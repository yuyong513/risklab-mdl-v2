"""建模数据解析：**从哪里读数据、用哪些列、按什么划分**。

★ 解析优先级（每一步都必须可见）
--------------------------------
| 顺序 | 数据来源 | 处理 |
|---|---|---|
| ① | ``preprocess/processed.parquet`` | 直接读（预处理阶段的产物，口径一致） |
| ② | ``preprocess/processed.csv`` | parquet 不可读（缺 pyarrow / 写失败）时的降级产物 |
| ③ | 原始 CSV + ``config.preprocess`` 规格 | 用预处理层的 ``validate_spec`` / ``apply_plan`` 在内存里跑一遍 |
| ④ | 原始 CSV | 文本/缺失**原样保留**，由模型层显式告警或报错 |

★ 为什么 ③ 不读 ``preprocess/*.json`` 重放
-------------------------------------------
``fill.json`` / ``encode.json`` 记录的是**执行结果**（每列做了什么、产出哪些列），
不是可执行规格（ordinal 的取值映射、WOE 表都没随产物落盘）。拿结果反推规格
会得到一个「看起来一样、实际不同」的处理链 —— 典型静默降级。
⇒ ③ 只认调用方显式给的 ``config.preprocess``；若只有 ``preprocess/*.json`` 而
processed 数据缺失，则**显式告警**并退回 ④（不假装处理过）。

★ 特征优先级
------------
``config.features`` → ``feature_selection/selected_features.txt`` → 全部数值列。
自动选择时非数值列**排除并告警**（列出前 10 个）；显式指定非数值列则**报错**
（用户点名要的列不能用，继续跑等于偷偷换特征）。

★ 划分优先级
------------
``split/split.json`` 的 spec 复算 mask → 无则内部 80/20 分层划分（**不落盘**，
告警明示「跨 run 复现请显式 dataset_split」）。processed 数据的行数与原 CSV 不一致
（预处理 drop 了行）时 split mask 无法对齐，同样退化为内部划分 + 告警。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..artifacts.manifest import ManifestError
from ..core.errors import ConfigError, DataError
from ..core.split import SplitSpec, split_frame
from ..data.loader import infer_roles, is_numeric_series, read_csv_auto
from ..platform.fs import read_json

__all__ = [
    "ModelFrame",
    "available_time_column",
    "load_model_frame",
    "resolve_features",
    "resolve_masks",
]

#: 预处理产物的候选文件（顺序即优先级）。
PROCESSED_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    ("preprocess/processed.parquet", "parquet"),
    ("preprocess/processed.csv", "csv"),
)

#: 内部划分的默认比例（train / val）与 test（不产生）。
INTERNAL_TRAIN_RATIO = 0.8


@dataclass
class ModelFrame:
    """已解析的建模数据框 + 来源说明（来源必须随产物回显）。"""

    frame: pd.DataFrame
    source: str
    warnings: List[str] = field(default_factory=list)
    processed_path: Optional[str] = None
    split_present: bool = False
    preprocess_spec_applied: bool = False


def _read_processed(layout, rel_path: str, fmt: str) -> pd.DataFrame:
    path = layout.project_dir / rel_path
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_model_frame(
    layout,
    manifest: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> ModelFrame:
    """按模块说明的 ①→④ 优先级取数。"""
    cfg = dict(config or {})
    warnings: List[str] = []
    target = (manifest.get("project") or {}).get("target")
    target_column = (target or {}).get("column") if isinstance(target, Mapping) else None

    # ---------- ① / ② 预处理产物 ----------
    for rel_path, fmt in PROCESSED_CANDIDATES:
        path = layout.project_dir / rel_path
        if not path.is_file():
            continue
        try:
            frame = _read_processed(layout, rel_path, fmt)
        except Exception as exc:  # noqa: BLE001 - 读不出来必须可见，然后试下一个
            warnings.append(
                f"{rel_path} 读取失败（{type(exc).__name__}: {exc}）：尝试下一个来源"
            )
            continue
        if frame.empty:
            warnings.append(f"{rel_path} 是空表（0 行）：尝试下一个来源")
            continue
        return ModelFrame(
            frame=frame,
            source=f"{rel_path}（预处理产物）",
            warnings=warnings,
            processed_path=rel_path,
            split_present=_split_doc(layout) is not None,
        )

    # ---------- ③ 原始 CSV + 显式预处理规格 ----------
    raw = _read_raw(layout, manifest)
    frame = raw.frame
    warnings.extend(raw.warnings)
    roles = _roles(manifest, frame)
    spec = cfg.get("preprocess")
    if spec:
        try:
            frame, spec_warnings = _apply_preprocess(
                frame, spec, target=target, roles=roles,
                has_split=_split_doc(layout) is not None,
            )
        except ConfigError as exc:
            raise ConfigError(
                f"config.preprocess 规格无法执行：{exc.message}"
            ) from exc
        warnings.extend(spec_warnings)
        warnings.append(
            "本次按 config.preprocess 在内存中应用预处理（未落盘）—— "
            "若要让后续 run 复用同一口径，请先调用 preprocess_apply"
        )
        return ModelFrame(
            frame=frame,
            source="原始 CSV + config.preprocess（内存应用，未落盘）",
            warnings=warnings,
            split_present=_split_doc(layout) is not None,
            preprocess_spec_applied=True,
        )

    # ---------- ④ 原始 CSV ----------
    pre_dir = layout.subdir("preprocess")
    has_specs = any(
        (pre_dir / name).is_file()
        for name in ("fill.json", "encode.json", "outlier.json", "manual_bins.json")
    )
    if has_specs:
        warnings.append(
            "preprocess/ 下有规格产物但缺少 processed.parquet|csv："
            "预处理的执行结果（ordinal 映射 / WOE 表）未随规格落盘，无法无损重放 ⇒ "
            "本次直接使用原始数据。请重跑 preprocess_apply 生成 processed 数据"
        )
    return ModelFrame(
        frame=frame,
        source="原始 CSV（未经过预处理产物）",
        warnings=warnings,
        split_present=_split_doc(layout) is not None,
    )


def _read_raw(layout, manifest: Mapping[str, Any]):
    csv = layout.csv_path
    if csv is None:
        raw = (manifest.get("project") or {}).get("csv_path")
        if not raw:
            raise ManifestError(
                f"manifest 未记录 csv_path，无法定位原始数据：{layout.project_dir}",
                code="manifest_schema_invalid",
                hint="重新执行 dataset_load 写入 csv_path",
            )
        csv = Path(str(raw))
    if not Path(csv).is_file():
        raise ManifestError(
            f"原始 CSV 不存在：{csv}",
            code="path_not_readable",
            hint="文件被移动/删除时，把它放回原位置或重跑 project_init",
        )
    return read_csv_auto(Path(csv))


def _apply_preprocess(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    target: Any,
    roles: Mapping[str, Any],
    has_split: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """用预处理层的 ``validate_spec`` + ``apply_plan`` 在内存里执行预处理规格。"""
    from ..preprocess.apply import apply_plan, validate_spec

    plan = validate_spec(spec, frame, target=target, roles=roles, has_split=has_split)
    outcome = apply_plan(frame, plan, target=target, split_present=has_split)
    return outcome.frame, list(outcome.warnings)


def _split_doc(layout) -> Optional[Mapping[str, Any]]:
    path = layout.subdir("split") / "split.json"
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except Exception:  # noqa: BLE001 - 损坏的划分文件不能用于取数，但上层要告警
        return None
    return doc if isinstance(doc, Mapping) else None


def _roles(manifest: Mapping[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    roles = (manifest.get("project") or {}).get("roles") or {}
    out = {key: roles.get(key) for key in ("id", "time", "split")}
    if not any(out.values()):
        inferred, _ = infer_roles(frame)
        out = {key: inferred.get(key) for key in ("id", "time", "split")}
    return out


def resolve_features(
    frame: pd.DataFrame,
    layout,
    manifest: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[str], str, List[str], List[str]]:
    """返回 ``(features, source, warnings, excluded_non_numeric)``。"""
    cfg = dict(config or {})
    warnings: List[str] = []
    roles = _roles(manifest, frame)
    target = (manifest.get("project") or {}).get("target") or {}
    target_column = target.get("column") if isinstance(target, Mapping) else None
    excluded = {
        str(name)
        for name in (
            target_column,
            roles.get("id"),
            roles.get("time"),
            roles.get("split"),
        )
        if name
    }
    columns = [str(c) for c in frame.columns]

    def _numeric_only(candidates: Sequence[str], *, strict: bool) -> Tuple[List[str], List[str]]:
        numeric: List[str] = []
        non_numeric: List[str] = []
        for name in candidates:
            if name not in frame.columns:
                continue
            if is_numeric_series(frame[name]):
                numeric.append(name)
            else:
                non_numeric.append(name)
        if non_numeric and strict:
            samples = {c: frame[c].dropna().astype(str).unique()[:3].tolist() for c in non_numeric[:3]}
            raise ConfigError(
                f"config.features 里点名了非数值列：{non_numeric[:10]}，样例：{samples}。"
                f"服务不会把它们当类别特征悄悄入模 —— 请先 preprocess_apply 编码，"
                f"或从 features 里去掉"
            )
        return numeric, non_numeric

    explicit = cfg.get("features")
    if explicit:
        names = [str(c) for c in explicit]
        unknown = [c for c in names if c not in columns]
        if unknown:
            raise ConfigError(
                f"config.features 里有不存在的列：{unknown[:10]}。"
                f"可用列示例：{columns[:20]}"
            )
        features, non_numeric = _numeric_only(names, strict=True)
        if not features:
            raise ConfigError("config.features 全部是非数值列，无法入模")
        return features, "config.features", warnings, non_numeric

    selected_path = layout.subdir("feature_selection") / "selected_features.txt"
    if selected_path.is_file():
        text = selected_path.read_text(encoding="utf-8")
        names = [line.strip() for line in text.splitlines() if line.strip()]
        present = [c for c in names if c in columns]
        missing = [c for c in names if c not in columns]
        if missing:
            warnings.append(
                f"selected_features.txt 里 {len(missing)} 个特征不在当前数据中，已略过："
                f"{missing[:10]}"
            )
        features, non_numeric = _numeric_only(present, strict=False)
        if non_numeric:
            warnings.append(
                f"{len(non_numeric)} 个非数值特征未入模（需先 preprocess_apply 编码）："
                f"{non_numeric[:10]}"
            )
        if features:
            return (
                features,
                "feature_selection/selected_features.txt",
                warnings,
                non_numeric,
            )
        warnings.append("selected_features.txt 没有可用数值特征：退回全部数值列")

    numeric = [
        c for c in columns
        if c not in excluded and is_numeric_series(frame[c])
    ]
    non_numeric = [
        c for c in columns
        if c not in excluded and not is_numeric_series(frame[c])
    ]
    if non_numeric:
        warnings.append(
            f"{len(non_numeric)} 个非数值列未入模（自动选择只取数值列）：{non_numeric[:10]}"
        )
    if not numeric:
        raise ConfigError(
            "当前数据没有任何可入模的数值列（目标/角色列已排除）。"
            "请先 preprocess_apply 编码，或用 config.features 指定"
        )
    return numeric, "数据中的全部数值列", warnings, non_numeric


def resolve_masks(
    layout,
    frame: pd.DataFrame,
    *,
    y_codes: pd.Series,
    target_column: str,
    seed: int = 42,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, pd.Series], Dict[str, Any], List[str]]:
    """返回 ``(masks, info, warnings)``；``masks`` 含 train/val/test 三个布尔 Series。"""
    cfg = dict(config or {})
    warnings: List[str] = []
    doc = _split_doc(layout)
    if doc and isinstance(doc.get("spec"), Mapping):
        try:
            result = split_frame(
                frame, SplitSpec.from_config(doc["spec"]), target=target_column
            )
        except Exception as exc:  # noqa: BLE001 - 复算失败必须可见
            warnings.append(
                f"split/split.json 无法复算（{type(exc).__name__}: {exc}）："
                f"改为模型层内部划分"
            )
        else:
            expected = sum(int(v) for v in (result.counts or {}).values())
            if expected != len(frame):
                warnings.append(
                    f"split.json 的行数（{expected}）与当前数据（{len(frame)}）不一致 —— "
                    f"预处理可能删过行；split mask 无法对齐，改为内部划分"
                )
            else:
                masks = {
                    name: result.mask(name).reindex(frame.index).fillna(False).astype(bool)
                    for name in ("train", "val", "test")
                }
                info = {
                    "source": "split/split.json",
                    "mode": result.spec.mode,
                    "fingerprint": result.fingerprint,
                    "counts": {k: int(v) for k, v in (result.counts or {}).items()},
                    "description": result.description,
                }
                if not bool(masks["train"].any()):
                    warnings.append("split.json 的 train 子集为空：改为内部划分")
                else:
                    return masks, info, warnings

    # ---------- 内部 80/20 分层划分 ----------
    masks, internal_warnings = _internal_split(frame.index, y_codes, seed=seed)
    warnings.extend(internal_warnings)
    warnings.append(
        "未使用 split/split.json ⇒ 模型层内部按 80/20 分层划分"
        f"（seed={seed}）。该划分**不落盘**：跨 run 复现请显式调用 dataset_split"
    )
    info = {
        "source": "模型层内部划分（80/20 分层，未落盘）",
        "mode": "internal_stratified",
        "fingerprint": None,
        "counts": {k: int(v.sum()) for k, v in masks.items()},
        "description": "80/20 分层（按标签）",
    }
    return masks, info, warnings


def _internal_split(
    index: pd.Index, y_codes: pd.Series, *, seed: int
) -> Tuple[Dict[str, pd.Series], List[str]]:
    warnings: List[str] = []
    rng = np.random.default_rng(int(seed))
    labels = y_codes.to_numpy()
    train_flags = np.zeros(len(index), dtype=bool)
    for label in np.unique(labels[~np.isnan(labels)]):
        positions = np.where(labels == label)[0]
        rng.shuffle(positions)
        n_train = max(1, int(round(len(positions) * INTERNAL_TRAIN_RATIO)))
        n_train = min(n_train, len(positions) - 1) if len(positions) > 1 else 1
        train_flags[positions[:n_train]] = True
    masks = {
        "train": pd.Series(train_flags, index=index, dtype=bool),
        "val": pd.Series(~train_flags, index=index, dtype=bool),
        "test": pd.Series(False, index=index, dtype=bool),
    }
    if not masks["train"].any():
        raise DataError("内部划分后 train 子集为空（样本量不足）")
    if not masks["val"].any():
        warnings.append("内部划分后 val 子集为空：早停不可用，将按固定轮数训练")
    return masks, warnings


def available_time_column(
    frame: pd.DataFrame, manifest: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
) -> Tuple[str, str]:
    """定位时间列，返回 ``(列名, 不可用原因)``；不可用时列名为空串。"""
    cfg = dict(config or {})
    candidate = (
        cfg.get("time_column")
        or cfg.get("month_column")
        or ((manifest.get("project") or {}).get("roles") or {}).get("time")
    )
    if not candidate:
        # manifest 里没有时间角色（project_init 之后未跑 dataset_load / 未确认角色）
        # ⇒ 现场推断一次，而不是直接跳过 PSI：静默跳过等于「假装做过了跨期检查」。
        inferred, _ = infer_roles(frame)
        candidate = inferred.get("time")
    if not candidate:
        return "", "未识别到时间列：PSI 跳过（不伪造数值）"
    candidate = str(candidate)
    if candidate not in frame.columns:
        return "", f"时间列 {candidate!r} 不在数据中：PSI 跳过"
    if pd.Series(frame[candidate]).nunique(dropna=True) < 2:
        return "", f"时间列 {candidate!r} 只有 1 个取值：无跨期可比，PSI 跳过"
    return candidate, ""
