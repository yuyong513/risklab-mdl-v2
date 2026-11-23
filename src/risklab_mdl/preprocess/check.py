"""入模前预处理检查（契约 §3.3 ``preprocess_check``）。

本模块回答一个**只看数据就能回答**的问题：

    「这批特征里，哪些不处理就进不了模型，哪些建议处理，各自的依据是什么？」

判定规则（全部可复现、可配置，见 ``DEFAULT_RULES`` / ``docs/preprocess-split.md``）
-------------------------------------------------------------------------------
====================================  ==========  ======================================
问题                                    级别         判据
====================================  ==========  ======================================
非数值列未编码                          must_fix    ``not is_numeric_series``（dtype ∈ categorical/text）
特征列有缺失                            must_fix    ``missing_rate > 0``（可用 ``missing_policy`` 降级）
高基数类别（取值太多，独热会爆炸）       suggested  非数值且 ``nunique > high_cardinality_max``（默认 20）或 dtype=text
缺失过多                                suggested   ``missing_rate >= high_missing_rate``（默认 0.5）
同值列（常量，IV 恒为 0）               suggested   ``nunique <= 1`` 或最高频占比 >= 0.999
极端值 / 偏态                           suggested   3σ 偏离 > 3 或 IQR 离群比例 > 1%（core 口径）
哨兵值冲突                              suggested   数值列真实取值里已出现 99999.0（sentinel 填充会混淆）
====================================  ==========  ======================================

★ 为什么「非数值未编码」是 must_fix 而「缺失」也默认为 must_fix
    这两类都会让**入模矩阵直接不可用**：非数值列无法进入 LR/GBDT 的数值矩阵；
    缺失在 LR 里会直接抛错（sklearn 拒绝 NaN），在 GBDT 里虽然能跑但语义不确定。
    「不报错但结果没法解释」比报错更糟，所以默认都标 must_fix；
    ``config.missing_policy="suggested"`` 可把它们降级为建议（并如实写进结果）。

★ 统计口径复用 ``core.transforms.profile_features``
    极端值/偏态/哨兵冲突的判据与 core 完全同一套（3σ 只标记、IQR 才做盖帽依据），
    避免「检查说没问题、执行时 core 说冲突」这种两套口径打架。

★ 失败必须可见
    ``profile_features`` 遇到不存在的列会抛错（不会静默跳过）；读表失败按三态返回；
    被检查的特征一律出现在 ``by_feature`` 里，即使没有任何问题（issues 为空列表）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .. import __version__
from ..artifacts.layout import resolve_project
from ..artifacts.manifest import now_iso, read_manifest
from ..core.transforms import SENTINEL_FILL_VALUE, profile_features
from ..data.loader import (
    infer_roles,
    infer_semantic_dtype,
    is_numeric_series,
    normalize_target,
    read_csv_auto,
    to_jsonable,
)
from ..platform.fs import atomic_write_json
from ..project import resolve_csv_path, roles_for_analysis
from ..protocol import (
    relative_paths,
    result_error,
    result_from_exception,
    result_needs_input,
    result_ok,
    target_question,
)

__all__ = [
    "CHECK_SCHEMA",
    "DEFAULT_RULES",
    "CheckResult",
    "check_frame",
    "preprocess_check",
    "select_feature_columns",
]

#: ``preprocess/check.json`` 的 schema 标识。
CHECK_SCHEMA = "risklab-mdl.preprocess-check/1"

#: 默认规则（``config`` 可逐项覆盖；全部写进产物，保证结论可复现）。
#:
#: * ``missing_policy``：缺失列的级别 ``must_fix`` / ``suggested`` / ``ignore``；
#: * ``high_missing_rate``：缺失率超过它 ⇒ 额外给「建议剔除/谨慎」提示；
#: * ``high_cardinality_max``：非数值列取值数超过它 ⇒ 建议 WOE/ordinal；
#: * ``same_value_rate``：最高频占比达到它 ⇒ 同值列（常量），建议剔除；
#: * ``sentinel_value``：哨兵值，真实取值里已出现即告警；
#: * ``sigma_multiple`` / ``iqr_factor`` / ``iqr_ratio``：与 core.transforms 同口径。
DEFAULT_RULES: Dict[str, Any] = {
    "missing_policy": "must_fix",
    "high_missing_rate": 0.5,
    "high_cardinality_max": 20,
    "same_value_rate": 0.999,
    "sentinel_value": SENTINEL_FILL_VALUE,
    "sigma_multiple": 3.0,
    "iqr_factor": 1.5,
    "iqr_ratio": 0.01,
}

_MUST_FIX = "must_fix"
_SUGGESTED = "suggested"


# --------------------------------------------------------------------------- #
# 特征列选择（tools/selection.py 复用同一份口径）
# --------------------------------------------------------------------------- #
def select_feature_columns(
    frame: pd.DataFrame,
    *,
    roles: Optional[Mapping[str, Any]] = None,
    target_column: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
) -> List[str]:
    """确定待处理特征列：显式列表优先，否则排除 id/time/split/target。

    Raises
    ------
    ValueError
        ``features`` 里含不存在的列（工具层会映射成 ``config_invalid``）。
    """
    all_columns = [str(c) for c in frame.columns]
    if features is not None:
        wanted = [str(f) for f in features]
        missing = [f for f in wanted if f not in all_columns]
        if missing:
            raise ValueError(
                f"config.features 中的列不存在：{missing[:10]}；"
                f"可用列示例：{all_columns[:10]}（共 {len(all_columns)} 列）"
            )
        return wanted
    excluded: set[str] = set()
    if target_column:
        excluded.add(str(target_column))
    for key in ("id", "time", "split"):
        value = (roles or {}).get(key)
        if value:
            excluded.add(str(value))
    return [name for name in all_columns if name not in excluded]


# --------------------------------------------------------------------------- #
# 检查结果
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    """一次预处理检查的结果。"""

    must_fix: List[str] = field(default_factory=list)
    suggested: List[str] = field(default_factory=list)
    by_feature: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        """落盘 ``preprocess/check.json`` 的内容（契约 §3.3 三键 + 明细）。"""
        return {
            "schema": CHECK_SCHEMA,
            "version": __version__,
            "generated_at": now_iso(),
            "features": list(self.features),
            "must_fix": list(self.must_fix),
            "suggested": list(self.suggested),
            "by_feature": to_jsonable(self.by_feature),
            "summary": to_jsonable(self.summary),
            "warnings": list(self.warnings),
        }

    def data_block(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "feature_count": len(self.features),
            "must_fix": list(self.must_fix),
            "suggested": list(self.suggested),
            "must_fix_count": len(self.must_fix),
            "suggested_count": len(self.suggested),
            "by_feature": to_jsonable(self.by_feature),
            "summary": to_jsonable(self.summary),
        }


def _issue(
    column: str,
    code: str,
    level: str,
    message: str,
    suggestion: str = "",
) -> Dict[str, str]:
    return {
        "column": column,
        "code": code,
        "level": level,
        "message": message,
        "suggestion": suggestion,
    }


def check_frame(
    frame: pd.DataFrame,
    *,
    target_column: Optional[str] = None,
    features: Sequence[str],
    roles: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> CheckResult:
    """对 ``features`` 逐列做入模前检查（纯计算，不落盘）。

    ``roles`` 只用于说明（哪些列是 id/time/split/target），不改变 ``features`` 的构成
    —— 特征清单由 :func:`select_feature_columns` 决定。
    """
    cfg = dict(config or {})
    rules = dict(DEFAULT_RULES)
    for key in DEFAULT_RULES:
        if key in cfg and cfg[key] is not None:
            rules[key] = cfg[key]

    missing_policy = str(rules["missing_policy"]).strip().lower()
    if missing_policy not in ("must_fix", "suggested", "ignore"):
        raise ValueError(
            f"config.missing_policy 只能是 must_fix / suggested / ignore，收到 {rules['missing_policy']!r}"
        )
    high_missing = float(rules["high_missing_rate"])
    cardinality_max = int(rules["high_cardinality_max"])
    same_value_rate = float(rules["same_value_rate"])
    sentinel_value = float(rules["sentinel_value"])

    names = [str(f) for f in features]
    if target_column is not None:
        # 显式 features 里误带了目标列时直接排除 —— 目标列永远不进预处理检查。
        names = [name for name in names if name != str(target_column)]
    if not names:
        raise ValueError("待检查特征为空（config.features 为空或全部被排除）")

    profiles = profile_features(
        frame,
        names,
        sigma_multiple=float(rules["sigma_multiple"]),
        iqr_factor=float(rules["iqr_factor"]),
        iqr_ratio=float(rules["iqr_ratio"]),
        sentinel_value=sentinel_value,
    )

    role_of: Dict[str, str] = {}
    for key in ("id", "time", "split"):
        value = (roles or {}).get(key)
        if value:
            role_of[str(value)] = key

    must_fix: List[str] = []
    suggested: List[str] = []
    by_feature: Dict[str, Dict[str, Any]] = {}
    code_counts: Dict[str, int] = {code: 0 for code in _ALL_CODES}
    n_rows = int(frame.shape[0])

    for name in names:
        series = frame[name]
        profile = profiles.profiles[name]
        dtype = infer_semantic_dtype(series)
        numeric = is_numeric_series(series)
        missing_rate = float(profile.missing_rate)
        n_missing = int(series.isna().sum())
        nunique = int(profile.n_unique)
        top_share = float(profile.top_value_share)
        issues: List[Dict[str, str]] = []

        # ① 非数值且未编码 —— 入模矩阵直接不可用
        if not numeric:
            high_card = bool(
                dtype == "text" or nunique > cardinality_max
            )
            issues.append(
                _issue(
                    name,
                    "not_numeric",
                    _MUST_FIX,
                    f"{name}: 非数值列（推断类型 {dtype}，取值 {nunique} 个）未编码，"
                    f"无法直接进入数值矩阵",
                    "spec.encode 中声明编码策略：取值少用 onehot，高基数用 ordinal/woe（woe 仅二分类）",
                )
            )
            if high_card:
                issues.append(
                    _issue(
                        name,
                        "high_cardinality",
                        _SUGGESTED,
                        f"{name}: 类别取值数 {nunique} 偏多（阈值 {cardinality_max}），"
                        f"独热会产生 {nunique} 列并带来稀疏共线",
                        "改用 ordinal / woe 编码，或用业务规则先合并低频取值",
                    )
                )

        # ② 缺失 —— 默认 must_fix（可配置降级）
        if missing_rate > 0:
            level = {
                "must_fix": _MUST_FIX,
                "suggested": _SUGGESTED,
            }.get(missing_policy, "")
            if level:
                issues.append(
                    _issue(
                        name,
                        "missing",
                        level,
                        f"{name}: 缺失 {missing_rate:.2%}（{n_missing}/{n_rows} 行），"
                        f"未处理时入模矩阵含 NaN",
                        "spec.missing 中声明策略：median/mean/mode/sentinel/as_bin/drop；"
                        "缺失本身若携带信息，用 as_bin 让它单独成箱",
                    )
                )
            if missing_rate >= high_missing:
                issues.append(
                    _issue(
                        name,
                        "high_missing",
                        _SUGGESTED,
                        f"{name}: 缺失率 {missing_rate:.2%} ≥ {high_missing:.0%}，"
                        f"填充后的信息量可能不足",
                        "考虑剔除该列，或改用 as_bin（缺失→独立箱）后观察 IV",
                    )
                )

        # ③ 同值列（常量）
        if nunique <= 1 or top_share >= same_value_rate:
            issues.append(
                _issue(
                    name,
                    "constant",
                    _SUGGESTED,
                    f"{name}: 同值列（取值 {nunique} 个，最高频占比 {top_share:.2%}），"
                    f"IV 恒为 0，对模型无贡献",
                    "spec.encode 中用 drop 剔除，或在特征清单里直接排除",
                )
            )

        # ④ 极端值 / 偏态（core 同一套判据）
        if numeric and profile.flag == "显著":
            sigma = profile.sigma_deviation
            iqr_ratio = profile.iqr_outlier_ratio
            issues.append(
                _issue(
                    name,
                    "extreme",
                    _SUGGESTED,
                    f"{name}: 极端值/偏态显著（3σ 偏离 {sigma}，IQR 离群比例 "
                    f"{iqr_ratio if iqr_ratio is not None else 0:.2%}）",
                    "spec.cap 中声明盖帽：absolute 给 min/max，quantile 给 lower/upper 分位",
                )
            )

        # ⑤ 哨兵值冲突
        if numeric and profile.has_conflicting_sentinel:
            issues.append(
                _issue(
                    name,
                    "sentinel_conflict",
                    _SUGGESTED,
                    f"{name}: 真实取值中已出现哨兵值 {sentinel_value:g}，"
                    f"用 sentinel 填充会让缺失与真实值不可区分",
                    "改用 mean/median/mode/as_bin，或换一个业务上不可能的哨兵值",
                )
            )

        for issue in issues:
            code_counts[issue["code"]] = code_counts.get(issue["code"], 0) + 1
            if issue["level"] == _MUST_FIX:
                must_fix.append(issue["message"])
            else:
                suggested.append(issue["message"])

        by_feature[name] = {
            "dtype": dtype,
            "numeric": bool(numeric),
            "role": role_of.get(name, "feature"),
            "nunique": nunique,
            "missing_count": n_missing,
            "missing_rate": round(missing_rate, 6),
            "same_value_rate": round(top_share, 6),
            "sigma_deviation": profile.sigma_deviation,
            "iqr_outlier_ratio": profile.iqr_outlier_ratio,
            "issues": issues,
            "must_fix": [i["message"] for i in issues if i["level"] == _MUST_FIX],
            "suggested": [i["message"] for i in issues if i["level"] == _SUGGESTED],
        }

    summary = {
        "rows": n_rows,
        "feature_count": len(names),
        "must_fix_count": len(must_fix),
        "suggested_count": len(suggested),
        "issues_by_code": {k: v for k, v in code_counts.items() if v},
        "rules": {
            "missing_policy": missing_policy,
            "high_missing_rate": high_missing,
            "high_cardinality_max": cardinality_max,
            "same_value_rate": same_value_rate,
            "sentinel_value": sentinel_value,
        },
    }
    warnings: List[str] = []
    if must_fix:
        warnings.append(
            f"{len(must_fix)} 条 must_fix（不处理则入模矩阵不可用）："
            "非数值未编码 / 缺失；见 preprocess/check.json"
        )
    if not must_fix and not suggested:
        warnings.append("未发现预处理问题：特征已可直接入模（仍需在训练期核对口径）")
    return CheckResult(
        must_fix=must_fix,
        suggested=suggested,
        by_feature=by_feature,
        features=names,
        summary=summary,
        warnings=warnings,
    )


#: 已知问题码（用于 summary 计数，保持即使为 0 也可见）。
_ALL_CODES: Tuple[str, ...] = (
    "not_numeric",
    "missing",
    "high_missing",
    "high_cardinality",
    "constant",
    "extreme",
    "sentinel_conflict",
)


# --------------------------------------------------------------------------- #
# 工具入口（tools/preprocess.py 复用）
# --------------------------------------------------------------------------- #
def preprocess_check(
    project_dir: str, config: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """入模前检查：读项目数据 + 目标，落 ``preprocess/check.json``。

    契约 §3.3：入参 ``project_dir``、``config?: {features?}``；
    产物 ``preprocess/check.json``：``{must_fix, suggested, by_feature}``。
    """
    try:
        cfg = dict(config or {})
        layout = resolve_project(project_dir)
        manifest = read_manifest(layout.project_dir)
        csv = resolve_csv_path(layout, manifest)

        target = normalize_target(cfg.get("target")) or normalize_target(
            (manifest.get("project") or {}).get("target")
        )

        frame_head = read_csv_auto(csv, nrows=200).frame
        all_columns = [str(c) for c in frame_head.columns]
        features = cfg.get("features")
        if features is not None:
            features = [str(f) for f in features]
            if not features:
                return result_error(
                    "features_empty",
                    "config.features 是空列表",
                    "留空（不传）表示检查全部特征（排除 id/time/split/target）",
                )
            unknown = [f for f in features if f not in all_columns]
            if unknown:
                return result_error(
                    "config_invalid",
                    f"config.features 中的列不存在：{unknown[:10]}",
                    f"可用列示例：{all_columns[:10]}（共 {len(all_columns)} 列）",
                )

        if target is None or target["column"] not in all_columns:
            reason = (
                "尚未确认目标字段。"
                if target is None
                else f"指定的目标列 {target['column']!r} 不在数据中。"
            )
            inferred, _ = infer_roles(frame_head)
            question, context = target_question(
                frame_head, inferred["target_candidates"], reason=reason
            )
            return result_needs_input(question, context)

        read = read_csv_auto(csv, usecols=None if features is None else sorted(
            {target["column"], *features}
        ))
        frame = read.frame
        warnings: List[str] = list(read.warnings)

        roles = roles_for_analysis(manifest, frame)
        try:
            names = select_feature_columns(
                frame,
                roles=roles,
                target_column=target["column"],
                features=features,
            )
        except ValueError as exc:
            return result_error(
                "config_invalid", str(exc), "按 docs/preprocess-split.md 检查 config.features"
            )

        result = check_frame(
            frame,
            target_column=target["column"],
            features=names,
            roles=roles,
            config=cfg,
        )
        warnings.extend(result.warnings)

        check_path = layout.subdir("preprocess") / "check.json"
        payload = result.to_payload()
        payload["target"] = to_jsonable(target)
        payload["roles"] = to_jsonable({k: roles.get(k) for k in ("id", "time", "split")})
        payload["csv_path"] = csv.as_posix()
        atomic_write_json(check_path, to_jsonable(payload))

        data = result.data_block()
        data["target"] = to_jsonable(target)
        data["project_dir"] = layout.project_dir.as_posix()
        data["check_path"] = layout.rel(check_path)
        return result_ok(
            data,
            artifacts=relative_paths(layout, [check_path]),
            warnings=warnings + [f"已写入 {layout.rel(check_path)}"],
        )
    except Exception as exc:  # noqa: BLE001 - 统一三态错误载荷
        return result_from_exception(exc)
