"""模型包：变换规格、填充规则、评分标尺的持久化与回放。

★ 落盘后按 manifest 校验，缺变换或缺标尺直接报错，不静默兜底。
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .errors import ModelError
from .encode import apply_encode_meta
from .metrics import Direction, ScoreScale
from .utils import round_value, same_label

__all__ = [
    "ModelPackage",
    "PACKAGE_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TransformSpec",
    "apply_transforms",
    "assert_score_monotonic",
    "check_scorable",
    "describe_scale",
    "ensure_numeric",
    "fill_missing",
    "load_package",
    "predict_proba_matrix",
    "predict_score",
    "save_package",
]

PACKAGE_SCHEMA_VERSION = "2.0"
"""模型包 schema 版本（**写出时**用的版本）。

版本不匹配时**拒绝加载**而不是尽力而为：分数标尺、变换语义、
填充值口径都可能跨版本变化，强行加载会产出「能跑但结论错」的结果。

* ``1.0`` —— 旧版口径：只有 ``lgb``，二分类，模型本身是 LightGBM 对象；
* ``2.0`` —— 当前口径：5 算法 × 二分类/多分类，新增
  ``task`` / ``num_class`` / ``classes`` / ``bad_class`` / ``algorithm_family`` /
  ``preprocess_spec``；模型对象是统一适配器（见 ``models.adapters``）。
  **1.0 的旧包仍可加载并打分**（见 :func:`predict_score` 的版本分支）——
  否则早期版本训练好的模型在升级后直接失效，属不可接受的回归。
"""

SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.0", "2.0")
"""可加载的 schema 版本（写出只会写 ``PACKAGE_SCHEMA_VERSION``）。"""


@runtime_checkable
class TransformSpec(Protocol):
    """推理期变换契约。

    实现必须是**纯数据 + 无外部依赖**：加载模型包的进程可能没有训练时的
    特征字典、编码表、甚至没有 LightGBM 本身。任何「训练期对象引用」
    都会让 pickle 失败或让加载后的行为不一致。
    """

    name: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...


def _to_numeric(series: pd.Series, column: str, context: str) -> pd.Series:
    """转数值，失败时给出可定位的报错。

    ★ 为什么必须显式转换而不是靠 ``astype(float)`` 硬转
    ------------------------------------------------------
    ``astype(float)`` 遇到无法转换的值会抛 ``ValueError: could not
    convert string to float``，消息里**不包含列名**。
    万级特征时，用户要自己加日志重跑一遍才能定位是哪一列。

    显式转换 + 带列名与策略名的报错，让问题一次定位。
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    try:
        return pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as exc:
        bad = series[pd.to_numeric(series, errors="coerce").isna() & series.notna()]
        preview = bad.value_counts().head(3).to_dict() if len(bad) else {}
        raise ModelError(
            f"{context}失败：列 {column!r} 存在无法转为数值的取值。"
            f"非数值样例：{preview}。"
            f"请检查该列的编码是否在预处理阶段完成 —— "
            f"模型包不带编码规格（encode_meta 为空）时不会在推理期隐式做编码"
            f"（训练/推理用不同口径是线上事故的主因）。"
        ) from exc


# --------------------------------------------------------------------------- #
# 模型包
# --------------------------------------------------------------------------- #
@dataclass
class ModelPackage:
    """模型包。落盘后即可独立完成打分，不依赖任何训练期上下文。"""

    model: Any
    algorithm: str
    """算法标识：``dt`` / ``lr`` / ``gbdt`` / ``xgb`` / ``lgb``（schema 2.0），
    schema 1.0 的旧包只有 ``lgb`` / ``lr`` / ``xgb`` 三种取值。"""
    features: list
    params: Mapping[str, Any] = field(default_factory=dict)
    best_iteration: int | None = None

    transforms: list = field(default_factory=list)
    """★ 旧版 LGB 包恒为空。后续接 LR 时写入符号翻转变换。"""

    fill_values: Mapping[str, float] = field(default_factory=dict)
    """按列的填充值。LGB 原生支持 NaN，通常可为空。"""

    score_scale: ScoreScale = field(default_factory=ScoreScale)
    bin_specs: Mapping[str, Any] = field(default_factory=dict)
    """分箱规格。旧版 LGB 包的分箱仅用于报告展示，但**仍要落盘**：
    增量重打分时的分箱必须与建模时完全一致，否则跨期分数不可比。"""

    direction_meta: Mapping[str, str] = field(default_factory=dict)
    """特征的业务方向声明。供报告核对。"""

    encode_meta: Mapping[str, Any] = field(default_factory=dict)
    """非数值特征编码规格（训练期捕获、随包落盘，推理期据此复现编码）。

    ★ 纯数据契约：strategy / applied / dropped / ordinal_maps /
    onehot_values / woe_tables / woe_fit_desc —— 全部为 dict/list/float/str，
    不含任何训练期对象引用（理由见 ``TransformSpec``）。

    为空时（旧包/无非数值特征）打分链路行为与 1.6.x 完全一致：
    输入必须是已编码的数值数据，缺列报错，绝不隐式编码。

    非空时，``predict_score`` 检测到入模特征缺失（即喂的是**原始数据**），
    会先按此规格显式编码再打分 —— 编码口径与训练期逐位一致
    （复现入口见 ``core.encode.apply_encode_meta``）。
    """

    label_spec_digest: str = ""

    feature_iv: Mapping[str, float] = field(default_factory=dict)
    """各特征的 IV（初筛阶段算的，随模型包一起传递，报告层直接引用不必重算）。"""

    feature_psi: Mapping[str, float] = field(default_factory=dict)
    """各特征的 PSI 聚合（取跨对比月的最大值 —— PSI 回答「有没有漂移」）。"""

    feature_psi_month: Mapping[str, str] = field(default_factory=dict)
    """PSI 最差月（与 ``feature_psi`` 对应）。"""

    feature_iv_bins: Mapping[str, Any] = field(default_factory=dict)
    """各特征 IV 计算时的（请求箱数, 实际箱数）。
    ★ IV 读数**依赖箱数** —— 不带这个参照系，报告里的 IV 列
      在跨项目横向比较时**没有参照系**。"""

    task: str = "binary"
    """任务类型（schema 2.0）：``binary`` / ``multiclass``。

    schema 1.0 的旧包没有这个字段 ⇒ 默认 ``binary``，与旧版口径一致。
    """

    num_class: int = 2
    """类别数（schema 2.0）。二分类恒为 2。"""

    classes: list = field(default_factory=list)
    """原始类别列表（顺序即编码，``classes[i]`` 的编码是 ``i``）。

    ★ 为什么原始类别必须随包落盘
    ---------------------------
    多分类打分要输出「每类的概率列」，列名只能是原始类别（``M0/M1/M2``）。
    若只存编码，推理期就不知道 ``proba_2`` 对应哪个业务类别 ——
    报表里的「M2 概率」与模型实际的第 3 列是否同义，无法验证。

    空列表表示「按 ``0..num_class-1`` 编码」（schema 1.0 旧包、或纯数值标签）。
    """

    bad_class: Any = None
    """「坏」类别的**原始取值**（二分类取正类概率；多分类据此选分数列）。

    缺省（``None``）时按 :meth:`bad_index` 的规则兜底：二分类取第 2 列，
    多分类取最后一类。
    """

    algorithm_family: str = ""
    """算法所属库（``sklearn`` / ``xgboost`` / ``lightgbm``），日志与报表用。"""

    preprocess_spec: Mapping[str, Any] = field(default_factory=dict)
    """预处理规格引用（``preprocess/summary.json`` 的关键字段快照）。

    ★ 存「引用 + 快照」而不是整份规格：打分链路只依赖 ``encode_meta`` /
      ``fill_values``（已各自落盘），这里记录的是「这份模型是在哪套预处理
      口径下训出来的」，供报告与排障核对。
    """

    training: Mapping[str, Any] = field(default_factory=dict)
    """训练期诊断信息（样本量、耗时等），仅供报表。"""

    metrics: Mapping[str, float] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = PACKAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.features:
            raise ModelError("ModelPackage.features 为空，无法打分")
        if len(set(self.features)) != len(self.features):
            dupes = sorted({f for f in self.features if list(self.features).count(f) > 1})
            raise ModelError(f"ModelPackage.features 存在重复特征名：{dupes}")
        unknown = set(self.fill_values) - set(self.features)
        if unknown:
            raise ModelError(
                f"fill_values 声明了模型包未使用的列：{sorted(unknown)}。"
                f"这是配置错误 —— 填充值只对入模特征有意义。"
            )
        if not isinstance(self.score_scale, ScoreScale):
            raise ModelError(
                f"score_scale 必须是 ScoreScale 实例，当前 {type(self.score_scale).__name__}。"
                f"裸 dict 会在概率转分数时静默用到默认标尺。"
            )
        # ---- schema 2.0 的字段校验（1.0 旧包用默认值，天然通过） ---- #
        if self.task not in ("binary", "multiclass"):
            raise ModelError(
                f"ModelPackage.task 必须是 'binary' / 'multiclass'，当前 {self.task!r}"
            )
        try:
            num_class = int(self.num_class)
        except (TypeError, ValueError) as exc:
            raise ModelError(
                f"ModelPackage.num_class 必须是整数，当前 {self.num_class!r}"
            ) from exc
        if num_class < 2:
            raise ModelError(f"ModelPackage.num_class 至少为 2，当前 {num_class}")
        if self.task == "binary" and num_class != 2:
            raise ModelError(
                f"二分类包的 num_class 必须是 2，当前 {num_class} —— "
                f"任务类型与类别数矛盾，概率列顺序无法确定"
            )
        if self.classes:
            if len(self.classes) != num_class:
                raise ModelError(
                    f"classes 长度({len(self.classes)}) 与 num_class({num_class}) 不一致"
                )
            if self.bad_class is not None and not any(
                same_label(self.bad_class, c) for c in self.classes
            ):
                raise ModelError(
                    f"bad_class={self.bad_class!r} 不在 classes={list(self.classes)} 中 —— "
                    f"分数口径无定义"
                )

    # ------------------------------------------------------------------ #
    @property
    def is_v2(self) -> bool:
        """是否是 schema 2.0 包（决定打分链路走新口径还是兼容口径）。"""
        return str(self.schema_version) not in ("1.0", "")

    def bad_index(self) -> int:
        """「坏类」在概率矩阵里的列号。"""
        if not self.classes:
            return 1 if int(self.num_class) >= 2 else 0
        if self.bad_class is not None:
            for index, label in enumerate(self.classes):
                if same_label(self.bad_class, label):
                    return index
            raise ModelError(
                f"bad_class={self.bad_class!r} 不在 classes={list(self.classes)} 中"
            )
        return len(self.classes) - 1

    # ------------------------------------------------------------------ #
    @property
    def has_transforms(self) -> bool:
        return bool(self.transforms)

    def transform_names(self) -> list:
        return [getattr(t, "name", type(t).__name__) for t in self.transforms]

    def digest(self) -> str:
        """模型包内容摘要（不含模型对象本身）。

        用于：① 校验两次训练是否产出了同一个包；② 在模型登记表里标识版本。
        **刻意不含 ``created_at``** —— 否则每次加载摘要都不同，失去比对意义。
        """
        payload = repr(
            (
                self.algorithm,
                list(self.features),
                dict(sorted(self.params.items())),
                self.transform_names(),
                sorted(self.fill_values.items()),
                self.score_scale.to_spec(),
                self.label_spec_digest,
                dict(sorted((getattr(self, "encode_meta", None) or {}).items())),
                # schema 2.0：任务/类别/坏类也进摘要 —— 同一份模型换一套标签编码
                # 就是「另一个模型」，摘要必须变，否则两次训练会被误判为同一个包
                self.task,
                list(self.classes),
                str(self.bad_class),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def manifest(self) -> dict:
        """人类可读的清单（不含模型对象）。用于产出物登记与核对。"""
        _em = getattr(self, "encode_meta", None) or {}
        return {
            "algorithm": self.algorithm,
            "algorithm_family": self.algorithm_family,
            "task": self.task,
            "num_class": int(self.num_class),
            "classes": list(self.classes),
            "bad_class": self.bad_class,
            "n_features": len(self.features),
            "features": list(self.features),
            "params": dict(self.params),
            "best_iteration": self.best_iteration,
            "transforms": self.transform_names(),
            "n_fill_values": len(self.fill_values),
            "score_scale": self.score_scale.to_spec(),
            "n_bin_specs": len(self.bin_specs),
            "encode_meta": {
                "strategy": str(_em.get("strategy", "") or ""),
                "n_applied": len(_em.get("applied") or []),
            } if _em else {},
            "label_spec_digest": self.label_spec_digest,
            "metrics": {k: round_value(v) for k, v in self.metrics.items()},
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "digest": self.digest(),
        }


# --------------------------------------------------------------------------- #
# 持久化
# --------------------------------------------------------------------------- #
def save_package(pkg: ModelPackage, path) -> Path:
    """Side effects: 写出模型包（pickle）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as fh:
            pickle.dump(pkg, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PicklingError) as exc:
        raise ModelError(
            f"模型包写出失败：{target} —— {exc}。"
            f"若报错含「can't pickle」，通常是 transforms 里放了训练期的对象"
            f"（如 DataFrame、sklearn Pipeline）—— transforms 必须是纯数据。"
        ) from exc
    return target


def _backfill_legacy_fields(pkg: ModelPackage) -> None:
    """给旧 pickle 补齐 schema 2.0 新增的字段。

    ★ 为什么必须补
    --------------
    pickle 只恢复实例 ``__dict__`` 里**当时存在**的字段。评估期为 2.0 新增的
    ``task`` / ``classes`` / ``bad_class`` 在旧包里根本没有对应属性，
    且 ``field(default_factory=...)`` 的默认值**不是类属性** ⇒
    直接访问 ``pkg.classes`` 会抛 ``AttributeError``（症状是「旧模型包打不开」）。
    ⇒ 加载后按 dataclass 的默认值补齐，缺字段一律回落默认（等价于 1.0 语义）。
    """
    import dataclasses

    for field_info in dataclasses.fields(pkg):
        if hasattr(pkg, field_info.name):
            continue
        if field_info.default is not dataclasses.MISSING:
            setattr(pkg, field_info.name, field_info.default)
        elif field_info.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            setattr(pkg, field_info.name, field_info.default_factory())  # type: ignore[misc]


def load_package(path) -> ModelPackage:
    """加载模型包，并校验 schema 版本与完整性。"""
    src = Path(path)

    if not src.exists():
        raise ModelError(
            f"模型包不存在：{src}。增量打分的输入必须是训练时产出的 model.pkl，"
            f"不能用本项目重新训练一份 —— 训练环境与推理环境不同，"
            f"重新训练得到的分数与当初的不可比。"
        )
    try:
        with src.open("rb") as fh:
            pkg = pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
        raise ModelError(
            f"模型包反序列化失败：{src} —— {exc}。"
            f"常见原因：① 文件损坏；② 加载环境的包版本与训练时不同"
            f"（LightGBM 跨大版本不保证可加载）。"
        ) from exc

    if not isinstance(pkg, ModelPackage):
        raise ModelError(
            f"{src} 不是合法的 ModelPackage（实际是 {type(pkg).__name__}）。"
            f"不要用 pickle 直接加载裸模型对象做打分 —— "
            f"那样会跳过变换、填充与标尺，结果不可复现。"
        )
    _backfill_legacy_fields(pkg)
    if pkg.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ModelError(
            f"模型包 schema 版本不支持：文件 {pkg.schema_version}，"
            f"当前代码支持 {list(SUPPORTED_SCHEMA_VERSIONS)}"
            f"（写出时用 {PACKAGE_SCHEMA_VERSION}）。请使用配套版本加载。"
        )
    if not pkg.features:
        raise ModelError(f"模型包 {src} 的 features 为空，无法打分")
    return pkg


# --------------------------------------------------------------------------- #
# 打分链路（顺序不可调换）
# --------------------------------------------------------------------------- #
def apply_transforms(df: pd.DataFrame, pkg: ModelPackage) -> pd.DataFrame:
    """按落盘顺序应用全部推理期变换。**打分的唯一入口，禁止旁路。**"""
    out = df
    for t in pkg.transforms:
        out = t.apply(out)
    return out


def fill_missing(X: pd.DataFrame, pkg: ModelPackage) -> pd.DataFrame:
    """按落盘填充值补缺失。

    缺省填充值未声明时**保留 NaN**（LightGBM 原生支持缺失分支）。
    ⚠ LR / XGBoost 不支持 NaN，后续接入时必须确保 ``fill_values`` 覆盖全部特征，
    否则模型会静默把 NaN 当 0 预测 —— 分数偏高且无任何提示。
    """
    if not pkg.fill_values:
        return X
    overlap = {k: v for k, v in pkg.fill_values.items() if k in X.columns}
    return X.fillna(overlap)


def check_scorable(df: pd.DataFrame, pkg: ModelPackage) -> None:
    """打分前的输入校验。**缺失列一律报错，绝不静默补 0。**"""
    missing = [c for c in pkg.features if c not in df.columns]
    if missing:
        preview = missing[:10]
        raise ModelError(
            f"打分数据缺少 {len(missing)} 个入模特征：{preview}"
            f"{' ...' if len(missing) > 10 else ''}。"
            f"服务**不会**用 0 补齐缺失列 —— 那会让这些特征贡献一个"
            f"凭空捏造的中性值，且完全无提示。"
        )
    if df.empty:
        raise ModelError("打分数据为空")


def predict_score(df: pd.DataFrame, pkg: ModelPackage) -> pd.DataFrame:
    """统一打分入口：取列 → 变换 → 填充 → 预测 → 标尺转换。

    顺序不可调换，理由见模块说明。

    ★ 编码复现（encode_meta 非空时）
    --------------------------------
    喂入的数据缺入模特征 ⇒ 视为**原始未编码数据**，
    先按落盘的 ``encode_meta`` 规格显式编码（训练/推理同一批 apply 函数），
    编码后仍缺列才报错。``encode_meta`` 为空时行为与旧版完全一致。
    """
    meta = getattr(pkg, "encode_meta", None) or {}
    if meta:
        # 「喂的是原始数据」的两种形态，都要触发编码：
        # ① onehot 改列名 ⇒ 入模特征缺列；
        # ② ordinal / WOE **原地改值** ⇒ 列都在、但已编码列仍是字符串。
        applied = set(meta.get("applied") or [])
        missing = [c for c in pkg.features if c not in df.columns]
        nonnumeric_applied = [
            c for c in pkg.features
            if c in df.columns and c in applied
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if missing or nonnumeric_applied:
            df = apply_encode_meta(df, meta)
            still = [c for c in pkg.features if c not in df.columns]
            if still:
                raise ModelError(
                    f"按模型包编码规格（strategy={meta.get('strategy')!r}）"
                    f"编码后仍缺少 {len(still)} 个入模特征：{still[:10]}。"
                    f"编码规格无法补出这些列 —— 请核对打分数据是否为训练同源的原始数据。"
                )

    check_scorable(df, pkg)

    X = df[pkg.features]
    X = apply_transforms(X, pkg)
    X = fill_missing(X, pkg)
    X = ensure_numeric(X)

    out = pd.DataFrame(index=df.index)
    if pkg.is_v2:
        matrix, classes = predict_proba_matrix(df, pkg)
        # 「坏类」列一律走 bad_index()：二分类也**不**硬编码第 2 列 ——
        # classes 顺序不是 [good, bad] 时，硬编码会让分数方向反转
        # （KS/AUC 仍能算出来，只是把好客户排到了最差）。
        bad_index = pkg.bad_index()
        if pkg.task == "binary":
            # 二分类列名与 schema 1.0 完全一致（下游报表/脚本不必改）
            prob = matrix[:, bad_index] if matrix.shape[1] > bad_index else matrix.ravel()
            out["predict_proba"] = prob
            out["fico_score"] = pkg.score_scale.prob_to_score(prob)
            return out
        # 多分类：先给「坏类概率 + 分数」（与 binary 同名的两列），再给逐类概率
        bad_prob = matrix[:, bad_index]
        out["predict_proba"] = bad_prob
        out["fico_score"] = pkg.score_scale.prob_to_score(bad_prob)
        for index, label in enumerate(classes):
            out[f"proba_{label}"] = matrix[:, index]
        return out

    # schema 1.0：旧版口径（只有二分类，模型对象自带 predict_proba / predict）
    prob = _predict_proba(X, pkg)
    out["predict_proba"] = prob
    out["fico_score"] = pkg.score_scale.prob_to_score(prob)
    return out


def predict_proba_matrix(
    df: pd.DataFrame, pkg: ModelPackage
) -> tuple[np.ndarray, list]:
    """概率矩阵 ``(n, k)`` + 类别顺序（**多分类打分的唯一入口**）。

    ★ 为什么要单独一个函数
    ----------------------
    ``predict_score`` 的输出是「给人看的数据框」，而指标计算要的是「(n, k) 矩阵」。
    若指标层自己再去调一次 ``model.predict_proba``，两处对「列顺序/是否取第 1 列」
    的判断就有两份实现 —— 多分类下极易一边用 OVR 列、一边用坏类列，
    指标看着有数、结论完全错位。

    Returns
    -------
    (matrix, classes)
        ``matrix[:, i]`` 是 ``classes[i]`` 的概率；schema 1.0 包返回
        ``(["0", "1"])`` 与两列（``1 - p, p``）。
    """
    X = df[pkg.features]
    X = apply_transforms(X, pkg)
    X = fill_missing(X, pkg)
    X = ensure_numeric(X)

    if not pkg.is_v2:
        prob = _predict_proba(X, pkg)
        return np.column_stack([1.0 - prob, prob]), [0, 1]

    model = pkg.model
    if hasattr(model, "predict_proba"):
        # 统一传「已变换 + 已填充」的 DataFrame：适配器按特征名取列，
        # 因此填充值（LR/DT/GBDT 必需）不会在这一步被绕过
        raw = model.predict_proba(X)
        matrix = np.asarray(raw, dtype=float)
        if matrix.ndim == 1:
            if int(pkg.num_class) != 2:
                raise ModelError(
                    f"多分类模型返回一维概率（{matrix.shape}）—— 无法确定列顺序，拒绝打分"
                )
            matrix = np.column_stack([1.0 - matrix, matrix])
    else:  # pragma: no cover - 兼容只有 predict 的模型对象
        prob = np.asarray(model.predict(X)).ravel()
        matrix = np.column_stack([1.0 - prob, prob])

    num_class = int(pkg.num_class)
    if matrix.ndim != 2 or matrix.shape[1] != num_class:
        raise ModelError(
            f"模型输出的概率矩阵形状 {matrix.shape} 与 num_class={num_class} 不一致 —— "
            f"模型与模型包的标签编码不匹配"
        )
    if matrix.shape[0] != len(df):
        raise ModelError(
            f"预测行数 {matrix.shape[0]} 与输入行数 {len(df)} 不一致"
        )
    if not np.isfinite(matrix).all():
        raise ModelError("概率矩阵中存在 NaN/Inf（检查特征值域与填充值）")
    classes = list(pkg.classes) if pkg.classes else list(range(num_class))
    return matrix, classes


def ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """确保入模特征全部是数值列。

    ★ 为什么要显式校验，而不是交给 LightGBM 自己判断
    --------------------------------------------------
    LightGBM 对 ``object`` / ``category`` 列会**当作类别特征处理**。
    于是「本该是数值、但因为某次上游变更变成了字符串」的这一列，
    会被安静地当成类别切分点使用 ——
    **模型能训练、能打分、报表全绿，只是这一个特征的用法完全变了。**

    缺失值（NaN）不在此列 —— 它是 LightGBM 的合法输入。
    整列全缺失的列也放行 —— LightGBM 会走默认分支。
    """
    bad = [
        c for c in X.columns
        if not pd.api.types.is_numeric_dtype(X[c]) and not X[c].isna().all()
    ]
    if bad:
        samples = {c: X[c].dropna().astype(str).unique()[:3].tolist() for c in bad[:3]}
        raise ModelError(
            f"入模特征中存在非数值列 {bad[:10]}"
            f"{' ...' if len(bad) > 10 else ''}，样例：{samples}。\n"
            f"服务**不会**把它当类别特征交给 LightGBM —— 那样模型能跑通、"
            f"报表也全绿，但这个特征的实际用法已经变了（数值切分 → 类别切分）。\n"
            f"请检查：① 上游数据是否带了千分位/单位；② 缺失是否被填成了字符串；"
            f"③ 编码是否在预处理阶段完成。"
        )
    return X


def _predict_proba(X: pd.DataFrame, pkg: ModelPackage) -> np.ndarray:
    """统一取概率。

    ★ LightGBM Booster 没有 ``predict_proba``，只有 ``predict``
    --------------------------------------------------
    直接 ``model.predict(X)`` 在 ``objective="binary"`` 下返回的**就是概率**
    （LGBM 会自动 sigmoid），所以这里不能因为「有 predict_proba 就优先用它」
    而在两个分支间产生口径差异。

    另一个易错点：``predict_proba`` 返回 ``(n, 2)`` 二维，必须取第 1 列。
    拿到 ``(n,)`` 的整体数组（AUC 算出来正好 0.5）是最难发现的形态 ——
    分数能算出来、报表能出，只有 KS/AUC 悄悄变成 0.5。
    """
    try:
        if hasattr(pkg.model, "predict_proba"):
            raw = pkg.model.predict_proba(X)
            arr = np.asarray(raw)
            if arr.ndim == 2:
                if arr.shape[1] < 2:
                    raise ModelError(
                        f"模型 predict_proba 返回 {arr.shape}，"
                        f"少于两列（应为 [P(0), P(1)]）。模型可能未正确加载。"
                    )
                prob = arr[:, 1]
            elif arr.ndim == 1:
                prob = arr
            else:
                raise ModelError(f"predict_proba 返回维度异常：{arr.shape}")
        else:
            prob = np.asarray(pkg.model.predict(X)).ravel()
    except ModelError:
        raise
    except Exception as exc:  # 依赖库抛的异常类型不稳定，统一包装
        raise ModelError(f"模型预测失败：{type(exc).__name__}: {exc}") from exc

    prob = np.asarray(prob, dtype=float).ravel()
    if prob.size != len(X):
        raise ModelError(
            f"预测结果长度 {prob.size} 与输入行数 {len(X)} 不符 —— "
            f"模型对象与特征列表不匹配（最常见：加载了别的模型包）。"
        )
    bad = ~np.isfinite(prob)
    if bad.any():
        raise ModelError(
            f"预测结果中有 {int(bad.sum())} 个非有限值（nan/inf）。"
            f"通常来自：特征含 inf、或模型未覆盖到某些取值。"
            f"不要静默替换为 0 —— 那会让这批样本的分数变成最好/最差。"
        )
    return prob


def describe_scale(pkg: ModelPackage) -> dict:
    """评分标尺的白话说明，供弹窗/报告呈现。"""
    spec = pkg.score_scale
    direction_text = (
        "分数越高风险越高" if spec.direction == "higher_score_higher_risk"
        else "分数越高风险越低"
    )
    return {
        "基准分": spec.base_point,
        "基准赔率": spec.base_odds,
        "PDO": spec.pdo,
        "分数区间": [spec.min_score, spec.max_score],
        "方向": direction_text,
        "含义": (
            f"分数每上升 {spec.pdo} 分，好坏赔率翻一倍。"
            f"基准分 {spec.base_point} 处赔率为 {spec.base_odds:g}。"
        ),
    }


def assert_score_monotonic(scores: Sequence[float], direction: Direction) -> None:
    """打分明细的顺序性自检。

    分数必须与风险**单调对应**。若发现反向，说明标尺方向配错或变换丢失 ——
    这类错误在任何单点指标上都不会暴露（KS 取了绝对值仍然「正常」），
    只有在按分数分段看坏样本率时才会显形。
    """
    if len(scores) < 2:
        return
    arr = np.asarray(scores, dtype=float)
    if not np.isfinite(arr).all():
        raise ModelError("打分明细中存在非有限分数")
    uniq = np.unique(arr)
    if len(uniq) < 2:
        raise ModelError(
            f"全部 {len(arr)} 个样本的分数完全相同（{uniq[0]:g}）。"
            f"这几乎必然是模型没学到东西，或打分链路丢失了变换。"
        )
