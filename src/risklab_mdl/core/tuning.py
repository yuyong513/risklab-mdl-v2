"""超参搜索空间与目标函数。

★ 目标函数权重支持 ks / auc / gap / lift 四项，权重不入搜索空间。
★ 权重为零的指标不参与打分，且必须记入 ignored 供报告呈现。
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ConfigError, ModelError
from .utils import round_value

__all__ = [
    "HPARAM_CATEGORIES",
    "HPARAM_CATEGORY_LABELS",
    "HPARAM_CATEGORY_MEMBERS",
    "WEIGHT_KEY_BY_METRIC",
    "ObjectiveWeights",
    "SearchSpace",
    "TuningConfig",
    "TuningResult",
    "TuningTrial",
    "assemble_space_from_categories",
    "build_search_space",
    "build_space_spec",
    "compose_space_spec",
    "tune_model",
    "validate_tune_on",
    "weights_from_metrics",
]

TuneOn = Literal["cv", "val", "train"]
"""允许的调参目标集。**没有 ``oot``** —— 这是结构性修复，
不是运行时报错，而是配置里压根不存在这个取值。"""


_ALLOWED_TUNE_ON: frozenset[str] = frozenset({"cv", "val", "train"})

_SPACE_TYPES: frozenset[str] = frozenset(
    {"choice", "uniform", "quniform", "loguniform", "qloguniform", "int"}
)


# --------------------------------------------------------------------------- #
# 目标函数权重
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObjectiveWeights:
    """目标函数权重。四项之和**不必**为 1 —— 只有相对大小有意义。

    ⚠ KS 与 AUC 同为**整体区分度**指标，相关系数通常 > 0.9。
    因此 ``ks=0.5, auc=0.4`` 接近「对单一指标加权」，
    两项相乘并不会得到「既看 KS 又看 AUC」的复合效果。

    若目的是**反映头部风险集中度**（风控真正在意的是最差 10% 客户），
    应把 ``auc`` 的权重挪给 ``lift`` —— 那才是与 KS 互补的维度。

    ★ 默认值与 risklab-mcp 一致（``lift=0``）——
      不改变既有口径，只是**开放**了调整的可能。
    """

    ks: float = 0.5
    auc: float = 0.4
    lift: float = 0.0
    """目标集 LIFT@pct 的权重。默认 0（不参与）。"""
    gap: float = 0.1
    """train 与 CV 目标集之间的衰减惩罚权重。"""

    def __post_init__(self) -> None:
        for name, value in (
            ("ks", self.ks),
            ("auc", self.auc),
            ("lift", self.lift),
            ("gap", self.gap),
        ):
            if value < 0:
                raise ConfigError(f"目标函数权重不允许为负：{name}={value}")
        if self.ks + self.auc + self.lift + self.gap <= 0:
            raise ConfigError("目标函数四项权重全为 0，目标函数无意义")

    def to_dict(self) -> dict[str, float]:
        return {
            "cv_ks": self.ks,
            "cv_auc": self.auc,
            "cv_lift": self.lift,
            "negative_train_cv_gap": self.gap,
        }


# --------------------------------------------------------------------------- #
# 搜索空间
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SearchSpace:
    """LightGBM 超参搜索空间。**全部来自流程定义，无类常量**。

    参数类型
    --------
    ``choice``
        ``{"type": "choice", "options": [2, 3, 4, 5]}``
    ``uniform`` / ``quniform``
        ``{"type": "quniform", "low": 20, "high": 300, "step": 10}``
    ``loguniform`` / ``qloguniform``
        同上，取值在对数尺度上均匀 —— 学习率这类跨数量级的参数用它更合适。
    ``int``
        ``{"type": "int", "low": 1, "high": 20}``

    ★ 为什么不做成类常量
    -------------------
    把搜索空间写成类常量的问题是**改它必须改代码**：不同业务（首贷 vs 复贷、
    不同额度段）合适的深度范围完全不同，写死在类里就等于「所有项目共用一套搜索空间」。
    本实现要求调用方显式传入，未声明的参数直接不参与搜索 ——
    少搜一个参数是安全的，静默用错一个范围不是。
    """

    params: Mapping[str, Mapping[str, Any]]
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.params:
            raise ConfigError(
                "搜索空间为空。每个超参都必须显式声明取值范围 —— "
                "服务不内置默认搜索空间（不同业务的合适范围差异极大，"
                "内置默认值等于替使用方做了业务判断）。"
            )
        for name, spec in self.params.items():
            if spec.get("type") not in _SPACE_TYPES:
                raise ConfigError(
                    f"参数 {name} 的类型 {spec.get('type')!r} 不支持，"
                    f"可选：{sorted(_SPACE_TYPES)}"
                )
            kind = spec["type"]
            if kind == "choice":
                if not spec.get("options"):
                    raise ConfigError(f"参数 {name} 是 choice 但未提供 options")
            else:
                for bound in ("low", "high"):
                    if bound not in spec:
                        raise ConfigError(f"参数 {name} 缺少 {bound}")
                if float(spec["high"]) < float(spec["low"]):
                    raise ConfigError(
                        f"参数 {name} 的 high({spec['high']}) 不能小于 low({spec['low']})"
                    )
                # ★ high == low 是**合法**的退化写法
                #   语义是「声明这个参数，但它不参与变化」。
                #   常见于把固定参数也放进空间，让调参与训练走同一条拼装路径。
                #   区别于「high < low」—— 那是写反了，必须报错。
                if kind == "quniform" and "step" not in spec:
                    raise ConfigError(f"参数 {name} 是 quniform，必须提供 step")
            if "log" in kind and float(spec["low"]) <= 0:
                raise ConfigError(
                    f"参数 {name} 使用对数尺度但 low={spec['low']} <= 0 —— "
                    f"对数尺度要求 low > 0"
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.params)

    def to_frame(self) -> pd.DataFrame:
        """空间声明表，用于弹窗呈现给使用方。

        ★ q 变体必须带上步长 —— 少了它，呈现出来的候选点密度是错的。
        """
        rows = []
        for name, spec in self.params.items():
            kind = spec["type"]
            if kind == "choice":
                declared = str(spec["options"])
            elif "log" in kind:
                declared = f"[{spec['low']}, {spec['high']}] (对数尺度)"
            else:
                declared = f"[{spec['low']}, {spec['high']}]"
            if kind.startswith("q") and "step" in spec:
                declared = f"{declared}，步长 {spec['step']}"
            rows.append(
                {"参数": name, "类型": kind, "搜索范围": declared}
            )
        return pd.DataFrame(rows, columns=["参数", "类型", "搜索范围"])


def build_search_space(spec: Mapping[str, Mapping[str, Any]], *, seed: int = 42) -> SearchSpace:
    """由流程定义里的 ``tuning.search_space`` 构造。"""
    if not isinstance(spec, Mapping):
        raise ConfigError("search_space 必须是映射（参数名 → 取值声明）")
    return SearchSpace(params=dict(spec), seed=seed)


# --------------------------------------------------------------------------- #
# 超参分类目录
# --------------------------------------------------------------------------- #
#: 四类超参的**唯一真源**：类别 → 该类下的参数名。
#:
#: ★ 为什么必须有这个目录
#: ---------------------
#: 用户要求「把模型参数进行分类（控制模型复杂度的、防止模型过拟合的…），
#: 然后按照不同类**逐次弹窗**」。要做到「逐类弹窗」，流程定义里就得知道
#: 哪个超参归哪一类。
#:
#: 而**流程定义是数据、代码是规则** —— 若这个目录写在流程 JSON 里，
#: 它就只是「一份待填的表」；若写在代码里，流程定义又得引用代码里的常量。
#: 两处各写一份必然漂移 ⇒ 归到**代码里**（core 层），
#: 流程定义通过**参数路径**引用它（如 ``tuning.space.complexity.max_depth``），
#: 执行层按同一份目录合并 ⇒ **弹窗与执行层永远看到同一套分类**。
#:
#: ★ 分类的业务含义
#: ------------------
#: ① complexity（控制模型复杂度）：树越多/越深 ⇒ 越复杂 ⇒ 越易过拟合
#: ② overfit（防止模型过拟合）：正则化与采样类参数，直接压制过拟合
#: ③ training（控制训练过程）：学习率、轮数、随机种子等
#: ④ objective（目标函数权重）：不是模型超参，而是**目标函数本身的取舍**
#:    —— 单列一类，避免用户把它误当成「模型参数」调
HPARAM_CATEGORIES: tuple[str, ...] = (
    "complexity",   # ① 控制模型复杂度
    "overfit",      # ② 防止模型过拟合
    "training",     # ③ 控制训练过程
    "objective",    # ④ 目标函数权重（不是模型超参）
)

HPARAM_CATEGORY_LABELS: dict[str, str] = {
    "complexity": "控制模型复杂度",
    "overfit": "防止模型过拟合",
    "training": "控制训练过程",
    "objective": "目标函数权重",
}

HPARAM_CATEGORY_MEMBERS: dict[str, tuple[str, ...]] = {
    # ① 控制模型复杂度：叶子数、深度
    #   ★ 不含 num_iterations —— 它与 training 类�� num_boost_round 语义重复
    #     （LightGBM 的 num_iterations 与 num_boost_round 是同一个东西），
    #     两个都放进搜索空间 ⇒ 优化器在两个同义参数间随机浪费试验次数。
    "complexity": ("num_leaves", "max_depth"),
    # ② 防止模型过拟合：叶子最小样本、正则、采样比例
    #   ★ 不含 min_child_weight —— 它与 min_data_in_leaf 高度相关，
    #     且 LightGBM 在 min_data_in_leaf 已设时默认不再用它（须显式 min_child_weight>0 才生效），
    #     放进搜索空间只会产出「配了却不生效」的参数。
    "overfit": (
        "min_data_in_leaf",
        "min_child_samples",
        "lambda_l1",
        "lambda_l2",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
    ),
    # ③ 控制训练过程：学习率、轮数、早停、种子
    "training": (
        "learning_rate",
        "num_boost_round",
        "early_stopping_rounds",
        "seed",
    ),
    # ④ 目标函数权重：cv_ks / cv_auc / cv_lift / negative_train_cv_gap
    "objective": ("cv_ks", "cv_auc", "cv_lift", "negative_train_cv_gap"),
}

#: ④ 目标函数权重项 → 内部权重键的映射。
#::
#: ★ 为什么需要这层映射
#: ---------------------
#: ``ObjectiveWeights`` 的字段名是 ``ks/auc/lift/gap``，
#: 但参数文档里的路径用的是**指标全名**（``cv_ks`` / ``cv_auc`` / …）——
#: 与 ``ObjectiveWeights.to_dict()`` 的键名一致，报表里才能对得上。
#:
#: ⇒ 目录里存**指标全名**，执行层据此翻译成权重对象字段。
WEIGHT_KEY_BY_METRIC: dict[str, str] = {
    "cv_ks": "ks",
    "cv_auc": "auc",
    "cv_lift": "lift",
    "negative_train_cv_gap": "gap",
}


def build_space_spec(params: Mapping[str, Any]) -> dict[str, Any]:
    """反向：把一组具体参数转成「取值固定」的空间声明。

    用途：最终训练时需要一份完整的参数集合（含未参与搜索的固定参数），
    而调参结果只包含搜索过的那些。把固定参数也表达成 ``choice``（只有一个选项）
    可以让两份参数走完全相同的拼装路径 ——
    **避免「训练用的参数」与「调参产出的参数」由两处代码分别拼装**，
    那正是参数不一致的根源。
    """
    return {
        name: {"type": "choice", "options": [value]}
        for name, value in params.items()
    }


def parse_space_spec(value: Any, *, name: str = "") -> dict[str, Any]:
    """把「一条超参的取值声明」规整成映射形态。

    Parameters
    ----------
    value
        两种形态都接受：

        * **映射**（dict）—— 直接使用。这是流程参数 ``default`` 的形态。
        * **JSON 文本**（str）—— 解析后使用。这是**弹窗 B 选项**让用户填的形态。

    name
        超参名，只用于报错信息定位。

    Returns
    -------
    ``{"type": ..., ...}`` —— 声明映射。

    ★★ 为什么要有这个函数（而不是只接受 dict）
    --------------------------------------------
    弹窗 B 选项引导用户输入的是**文本**（JSON 文本），而执行层要的是**映射**。
    两边形态不一致时，参数文档里存的是字符串 ⇒ 装配层拿到字符串 ⇒
    报「取值声明必须是映射」⇒ **走 B 路径的用户必然失败**。

    ⇒ 形态转换**收口在这里**：弹窗写什么，执行层怎么解析，只此一处定义。

    ★★ 为什么 JSON 文本要**校验**而不是直接 ``json.loads``
    ----------------------------------------------
    ``json.loads`` 对任意字符串都能抛或返回非 dict（如 ``"abc"`` 会被当裸串报错，
    ``"[1,2]"`` 会解析成 list）。⇒ 必须显式校验「必须是对象」，
    否则错误会以「后续字段缺失」的形式在别处爆出来 —— 又是一处「失败不可见」。
    """
    if isinstance(value, Mapping):
        return dict(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError(
                f"超参 {name or '(未命名)'} 的取值声明为空。"
                f"需要形如 {{\"type\": \"choice\", \"options\": [15, 31, 63]}} 的 JSON 文本。"
            )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"超参 {name or '(未命名)'} 的取值声明不是合法 JSON：{exc}。"
                f"需要形如 {{\"type\": \"choice\", \"options\": [15, 31, 63]}} 的 JSON 文本。"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise ConfigError(
                f"超参 {name or '(未命名)'} 的取值声明必须是 JSON 对象，"
                f"实际解析出 {type(parsed).__name__}。"
            )
        return dict(parsed)

    raise ConfigError(
        f"超参 {name or '(未命名)'} 的取值声明必须是映射或 JSON 文本，"
        f"实际为 {type(value).__name__}。"
    )


def compose_space_spec(
    default: Any,
    bounds: Any,
    *,
    name: str = "",
) -> dict[str, Any]:
    """把「字段位填写的边界数字」组装成完整的超参取值范围声明。

    ★★ 为什么需要这个入口（1.4.2 字段位输入）
    ----------------------------------------
    1.4.1 及以前，弹窗 B 选项要求用户在一个文本框里
    「按下列顺序填写（每项写成 JSON 文本）」共 N 项。实测三个后果：

      ① **靠位置对位**：少填一项 ⇒ 后面全部错位，且各声明本身都合法
         ⇒ ``SearchSpace`` 校验通过 ⇒ **静默串参数**（不报错、结果全错）；
      ② 用户要手写 ``{"type":"quniform","low":20,"high":300,"step":10}``
         —— 类型由用户填 ⇒ 整数参数写成 ``uniform`` 也过声明期校验，
         到采样后才被 LightGBM 拒绝；
      ③ 每项 JSON 内部自带逗号，而弹窗又承诺「逗号/顿号/｜均可分隔」
         ⇒ 切分边界只能靠人（或模型）猜。

    ⇒ 改为**每个超参一个字段位、按参数路径寻址、只填数字**：
      - ``type`` 从 ``default`` 取，用户**填不出**（② 结构性消失）；
      - 寻址靠路径，不靠位置（① 结构性消失）；
      - 用户只写数字，不写 JSON（③ 结构性消失）。

    Parameters
    ----------
    default
        流程定义里该参数的 ``spec.default``（完整声明，含 ``type``）。
        ★ 这是**类型与边界键的唯一真源**：``choice`` → ``options``，
          ``quniform`` → ``low/high/step``，``uniform/int`` → ``low/high``。
          新增一种分布类型时**不用改这里** —— 边界键从 default 的键推导。
    bounds
        字段位回填值，形如 ``{"low": 20, "high": 300}``（**不含 type**）。
        未出现的边界回落到 ``default``，因此「只改 1 项」是合法输入。
    name
        超参名，只用于报错信息定位。

    ★ 整数约束也从数据推导
    ----------------------
    ``default[key]`` 是 int（非 bool）⇒ 该边界只收整数；
    是 ``list[int]`` ⇒ 列表元素只收整数；是 float ⇒ 不作整数约束。
    ⇒ 不用维护「哪些 type 是整数类」的映射表（那表必然与 default 漂移）。

    ★ 组装完立刻过 ``SearchSpace`` 校验
    --------------------------------
    ``high < low`` / ``quniform`` 缺 ``step`` / ``choice`` 空 options /
    对数尺度 low≤0 都在**填写当场**报错，而不是等到调参采样才炸。
    """
    label = name or "(未命名)"
    if not isinstance(default, Mapping) or not default.get("type"):
        raise ConfigError(
            f"超参 {label} 的流程定义 default 不是完整声明（缺 type），"
            "无法按字段位组装取值范围。请检查 flows/*.json 的 params 声明。"
        )
    if not isinstance(bounds, Mapping):
        raise ConfigError(
            f"超参 {label} 的字段位取值必须是映射，实际为 {type(bounds).__name__}。"
            "字段位只填数字，例如 {\"low\": 20, \"high\": 300}。"
        )
    if "type" in bounds:
        raise ConfigError(
            f"超参 {label} 的字段位取值不允许带 type —— "
            f"类型固定为流程定义的 {default['type']!r}（避免整数参数被写成连续分布）。"
        )

    kind = default["type"]
    bound_keys = [k for k in default if k != "type"]
    unknown = [k for k in bounds if k not in bound_keys]
    if unknown:
        raise ConfigError(
            f"超参 {label} 收到未知边界 {unknown}；"
            f"{kind} 类型的可填边界只有 {bound_keys}。"
        )

    spec: dict[str, Any] = {"type": kind}
    for key in bound_keys:
        base = default[key]
        value = bounds.get(key, base)
        spec[key] = _coerce_bound(key, value, base, label)

    # ★ 组装完当场校验 —— 不必等到 assemble_search_space 那一步
    SearchSpace(params={label: spec})
    return spec


def _coerce_bound(key: str, value: Any, base: Any, label: str) -> Any:
    """按 ``default`` 里该边界的类型，规整一个字段位取值。"""
    int_like = isinstance(base, int) and not isinstance(base, bool)
    int_list = (
        isinstance(base, (list, tuple))
        and bool(base)
        and all(isinstance(v, int) and not isinstance(v, bool) for v in base)
    )
    if isinstance(base, (list, tuple)):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(
                f"超参 {label} 的 {key} 需要列表，实际为 {type(value).__name__}。"
            )
        items = list(value)
        if int_list:
            bad = [v for v in items if isinstance(v, bool) or not _is_integer(v)]
            if bad:
                raise ConfigError(
                    f"超参 {label} 的 {key} 只收整数，收到 {bad}。"
                )
            return [int(v) for v in items]
        return items
    if int_like:
        if isinstance(value, bool) or not _is_integer(value):
            raise ConfigError(
                f"超参 {label} 的 {key}={value!r} 不是整数；"
                "该参数按整数档位采样，写成小数会被 LightGBM 拒绝。"
            )
        return int(value)
    return value


def _is_integer(value: Any) -> bool:
    """整数或「整数值的浮点/数字串」都算整数；真小数不算。"""
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, str):
        try:
            return float(value.strip()).is_integer()
        except ValueError:
            return False
    return False


def assemble_space_from_categories(
    space_by_param: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """把「逐类弹窗」确认的超参范围合并成一份统一的搜索空间声明。

    Parameters
    ----------
    space_by_param
        ``{超参名: 取值声明}`` —— 每个超参一条，声明形如
        ``{"type": "choice", "options": [2, 3, 4, 5]}``。

    Returns
    -------
    ``{超参名: 取值声明}`` —— 与 :func:`build_search_space` 的入参同构。

    ★★ 为什么「合并」这件事要有显式实现，而不是靠调用方自己拼
    -----------------------------------------------------------
    4 个分类弹窗各自落盘了 4 组参数（``tuning.space.complexity.*`` 等），
    最终要凑成**一份**搜索空间。如果这件事由调用方（流程定义/执行层）
    临场拼，两处各拼一次就会漂移 ——
    「弹窗里显示的搜索范围」与「实际搜索的范围」不一致，
    而这种不一致**不报错**，只是搜出来的参数不是客户拍的那一档。

    ⇒ 合并逻辑收口在这里，弹窗与执行层都走这一条路径。

    ★★ 目标函数权重**不进搜索空间**
    ------------------------------
    ④ 类的 ``cv_ks`` / ``cv_auc`` / ``cv_lift`` / ``negative_train_cv_gap``
    是**目标函数的权重**，不是模型超参 —— 它们决定「怎么打分」，
    而不是「搜哪个参数」。把它们塞进搜索空间 ⇒ 优化器会去搜「权重」，
    语义完全错乱。

    ⇒ 本函数**显式剔除**这四项，由 :func:`weights_from_metrics` 单独处理。
    """
    out: dict[str, dict[str, Any]] = {}
    for name, spec in (space_by_param or {}).items():
        if name in WEIGHT_KEY_BY_METRIC:
            continue          # ★ 目标函数权重不是模型超参
        if not isinstance(spec, Mapping):
            raise ConfigError(
                f"超参 {name} 的取值声明必须是映射，实际为 {type(spec).__name__}"
            )
        out[name] = dict(spec)
    if not out:
        raise ConfigError(
            "搜索空间为空 —— 每个超参都必须显式声明取值范围。"
            "服务不内置默认搜索空间（不同业务的合适范围差异极大，"
            "内置默认值等于替使用方做了业务判断）。"
        )
    return out


def weights_from_metrics(metrics: Mapping[str, Any]) -> ObjectiveWeights:
    """由「指标全名 → 权重」的映射构造 :class:`ObjectiveWeights`。

    ★ 为什么不用 ``ObjectiveWeights(**metrics)``
    ---------------------------------------------
    指标全名（``cv_ks``）与权重字段名（``ks``）**不同名** ——
    前者要与报表里的指标名对齐，后者是内部字段。
    直接展开会把 ``cv_ks`` 当成未知字段传进 dataclass ⇒ ``TypeError``。

    ★ 未知指标名必须**报错**，不能静默忽略
    -------------------------------------------
    静默忽略的后果：用户把某项权重填错了名字 ⇒ 那项权重没生效 ⇒
    「目标函数和客户以为的不一样」，而报表里看不出任何异常。
    """
    kwargs: dict[str, float] = {}
    unknown: list[str] = []
    for metric, value in (metrics or {}).items():
        field = WEIGHT_KEY_BY_METRIC.get(str(metric))
        if field is None:
            unknown.append(str(metric))
            continue
        kwargs[field] = float(value)
    if unknown:
        raise ConfigError(
            f"未知的目标函数权重项：{unknown}。"
            f"可用项：{sorted(WEIGHT_KEY_BY_METRIC)}"
        )
    return ObjectiveWeights(**kwargs)


# --------------------------------------------------------------------------- #
# 调参配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TuningConfig:
    """调参配置。"""

    space: SearchSpace
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    n_iters: int = 100
    engine: Literal["hyperopt"] = "hyperopt"
    tune_on: TuneOn = "cv"
    n_splits: int = 5
    shuffle: bool = True
    gap_threshold: float = 0.05
    """train 与目标集的指标差距超过此值才施加惩罚。

    不设阈值地惩罚，等于对「目标集略差」的参数一视同仁地扣分 ——
    信贷样本量下 CV 每折几十个坏样本，波动 0.02 很正常，
    无阈值惩罚会把随机波动当成过拟合，压低整体目标值、扭曲参数排序。
    """
    failure_penalty: float = -999.0
    max_consecutive_failures: int = 10
    """熔断阈值。连续这么多次训练异常即中止，返回已成功的最好结果。"""
    num_boost_round: int = 1000
    early_stopping_rounds: int = 200
    lift_pct: float = 0.10
    """LIFT 取尾部的比例。"""
    lift_min_seg_size: int = 30
    """LIFT 段内样本量下限。低于此值只标记 ``is_reliable=False``，不静默修正。"""
    seed: int = 42
    fixed_params: Mapping[str, Any] = field(default_factory=dict)
    """不参与搜索的固定参数（如 ``objective="binary"``、``metric="auc"``）。"""

    def __post_init__(self) -> None:
        validate_tune_on(self.tune_on)
        if self.n_iters < 1:
            raise ConfigError(f"n_iters 至少为 1，当前 {self.n_iters}")
        if self.n_splits < 2 and self.tune_on == "cv":
            raise ConfigError(
                f"tune_on='cv' 时 n_splits 至少为 2，当前 {self.n_splits}。"
                f"单折不是交叉验证 —— 若只想跑一次，请显式设 tune_on='val'。"
            )
        overlap = set(self.space.names) & set(self.fixed_params)
        if overlap:
            raise ConfigError(
                f"以下参数同时出现在搜索空间与固定参数中：{sorted(overlap)}。"
                f"同一个超参既搜索又固定，配置自相矛盾。"
            )
        if self.gap_threshold < 0:
            raise ConfigError(f"gap_threshold 不允许为负，当前 {self.gap_threshold}")

    def derive(self, **overrides: Any) -> "TuningConfig":
        return replace_cfg(self, **overrides)


def validate_tune_on(value: str) -> None:
    """协议层禁用 OOT 调参。

    做成**结构性的** —— 配置里不存在 ``oot`` 这个取值，
    而不是「允许传入但运行时警告」。后者总有一天会被绕过。
    """
    if value not in _ALLOWED_TUNE_ON:
        raise ConfigError(
            f"tune_on={value!r} 非法。允许：{sorted(_ALLOWED_TUNE_ON)}。"
            f"**没有 'oot'** —— 用 OOT 选超参属于广义数据穿越"
            f"（等价于在测试集上调参，OOT 指标会系统性偏高、线上达不到）。"
            f"若确实需要外样本参与，请用 external_eval 做**事后**对比，"
            f"而不是让它参与参数选择。"
        )


def replace_cfg(cfg: TuningConfig, **overrides: Any) -> TuningConfig:
    """派生一个改了几项的新配置（frozen dataclass 用）。"""
    return dataclasses.replace(cfg, **overrides)


# --------------------------------------------------------------------------- #
# 试验轨迹
# --------------------------------------------------------------------------- #
@dataclass
class TuningTrial:
    """单次试验的完整记录。失败也记录 —— 静默跳过等于伪造轨迹。"""

    index: int
    params: Mapping[str, Any]
    status: Literal["ok", "failed"]
    target: float = float("nan")
    """目标函数值（**取负后**，与优化器约定一致）。``nan`` 表示未计算。"""
    metrics: Mapping[str, float] = field(default_factory=dict)
    error: str = ""
    elapsed_sec: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """转成报表行。

        ★ 参数列**不加前缀**
        --------------------
        早期实现给参数列加了 ``参数:`` 前缀来与指标列区分，结果是
        使用方在 Excel 里按 ``max_depth`` 排序/筛选时匹配不到任何一列。
        参数名本身就是唯一标识，不需要前缀。
        """
        return {
            "序号": self.index,
            "目标值": None if np.isnan(self.target) else round(self.target, 6),
            "状态": "成功" if self.status == "ok" else "失败",
            "耗时(秒)": round(self.elapsed_sec, 3),
            "KS": round_value(self.metrics.get("cv_ks")),
            "AUC": round_value(self.metrics.get("cv_auc")),
            "LIFT": round_value(self.metrics.get("cv_lift")),
            "train-AUC": round_value(self.metrics.get("train_auc")),
            "衰减": round_value(self.metrics.get("train_cv_gap_penalized")),
            "原始衰减": round_value(self.metrics.get("train_cv_gap")),
            "有效轮数": self.metrics.get("best_iteration"),
            "失败原因": self.error,
            **dict(self.params),
        }


@dataclass
class TuningResult:
    """调参结果 + 可复现轨迹。"""

    best_params: Mapping[str, Any]
    best_target: float
    trials: list[TuningTrial] = field(default_factory=list)
    best_metrics: Mapping[str, float] = field(default_factory=dict)
    aborted: bool = False
    abort_reason: str = ""
    engine: str = "hyperopt"
    seed: int = 42
    n_iters_requested: int = 0

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_failed(self) -> int:
        return sum(1 for t in self.trials if t.status == "failed")

    @property
    def success_rate(self) -> float:
        return 1.0 - self.n_failed / self.n_trials if self.trials else 0.0

    def trials_frame(self) -> pd.DataFrame:
        cols = [
            "序号", "目标值", "状态", "耗时(秒)", "KS", "AUC", "LIFT",
            "train-AUC", "衰减", "原始衰减", "有效轮数", "失败原因",
        ]
        if not self.trials:
            return pd.DataFrame(columns=cols)
        rows = [t.as_row() for t in self.trials]
        extra = [c for c in rows[0] if c not in cols]
        return pd.DataFrame(rows, columns=cols + extra)

    def summary(self) -> dict[str, Any]:
        return {
            "best_target": None if np.isnan(self.best_target) else round(self.best_target, 6),
            "best_params": dict(self.best_params),
            "best_metrics": {k: round_value(v) for k, v in self.best_metrics.items()},
            "n_trials": self.n_trials,
            "n_requested": self.n_iters_requested,
            "n_failed": self.n_failed,
            "success_rate": round(self.success_rate, 4),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "engine": self.engine,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------- #
# 目标函数
# --------------------------------------------------------------------------- #
def _objective_value(
    metrics: Mapping[str, float],
    weights: ObjectiveWeights,
    *,
    gap_threshold: float = 0.05,
) -> float:
    """按权重合成目标值。

    衰减项的处理：只有 ``train − CV目标`` **超过** ``gap_threshold`` 才全额惩罚，
    阈值以内不扣分 —— 理由见 :attr:`TuningConfig.gap_threshold`。

    ★ 读哪一列
    ----------
    优先读 ``train_cv_gap_penalized``（由 :func:`apply_gap_threshold` 按阈值裁剪过）；
    该列不存在（直接调用本函数）时才按 ``gap_threshold`` 就地裁剪。
    **绝不**直接按原始 ``train_cv_gap`` 扣分 —— 那会让「阈值以内不罚」的承诺失效。

    ★ LIFT 项的处理
    ---------------
    ``cv_lift`` 权重为 0 时**完全不读**该指标 ⇒
    「指标缺失」不会因为权重为 0 而让目标函数算不出来。
    权重 > 0 但指标缺失 ⇒ 目标函数算不出 ⇒ 该次试验判失败（**如实暴露**，
    不静默当 0 处理 —— 静默当 0 会让「LIFT 算不出来」变成「LIFT 很差」）。
    """
    ks = float(metrics.get("cv_ks", float("nan")))
    auc = float(metrics.get("cv_auc", float("nan")))
    raw_gap = float(metrics.get("train_cv_gap", 0.0))

    if np.isnan(ks) or np.isnan(auc):
        return float("nan")

    value = weights.ks * ks + weights.auc * auc

    if weights.lift > 0:
        lift = float(metrics.get("cv_lift", float("nan")))
        if np.isnan(lift):
            return float("nan")
        value += weights.lift * lift

    if "train_cv_gap_penalized" in metrics:
        penalized_gap = float(metrics["train_cv_gap_penalized"])
    else:
        penalized_gap = raw_gap if raw_gap > float(gap_threshold) else 0.0
    penalized_gap = penalized_gap if penalized_gap > 0 else 0.0
    return value - weights.gap * penalized_gap


def apply_gap_threshold(metrics: Mapping[str, float], threshold: float) -> dict[str, float]:
    """把未设阈值的衰减量按阈值裁剪。返回新 dict，不改入参。"""
    out = dict(metrics)
    gap = float(out.get("train_cv_gap", 0.0))
    out["train_cv_gap_penalized"] = gap if gap > threshold else 0.0
    return out


# --------------------------------------------------------------------------- #
# 调参主循环
# --------------------------------------------------------------------------- #
TrialFn = Callable[[dict[str, Any], int], tuple[Mapping[str, float], int | None]]
"""单次试验的求值函数。返回 ``(指标, 有效轮数)``；抛异常视为该组参数失败。"""


def tune_model(cfg: TuningConfig, trial_fn: TrialFn) -> TuningResult:
    """按配置执行超参搜索。

    Parameters
    ----------
    cfg
        调参配置（含搜索空间、目标函数权重、熔断策略）。
    trial_fn
        单次试验的求值函数。**由调用方注入** —— 本模块不关心用哪套指标、
        怎么训练，它只负责「怎么搜、失败怎么办、什么时候停」。

    Returns
    -------
    TuningResult
        最优参数 + 全部试验轨迹（含失败记录）。

    Notes
    -----
    ★ 目标值取负
    ----------
    本函数返回的 ``best_target`` 是**原始目标值**（越大越好）。
    内部传给 ``hyperopt.fmin`` 时取负（``hyperopt`` 只做最小化）。
    轨迹表里记录的也是原始值 —— 报表里「目标值越大越好」符合直觉。
    """
    if cfg.engine != "hyperopt":
        raise ConfigError(
            f"不支持的调参引擎：{cfg.engine!r}。唯一引擎是 'hyperopt'"
            f"（TPE 贝叶斯优化）。自研随机采样器已下线 —— "
            f"同样预算下样本利用率低、且两条实现需双份维护。"
        )
    trials = _search_hyperopt(cfg, trial_fn)

    ok = [t for t in trials if t.status == "ok" and not np.isnan(t.target)]
    if not ok:
        raise ModelError(
            f"调参未产生任何成功试验（共 {len(trials)} 次，成功 0 次）。"
            f"常见原因：搜索空间与数据量严重不匹配"
            f"（如 min_data_in_leaf 远大于样本数），或 LightGBM 不可用。"
            f"最近一次失败：{trials[-1].error if trials else '无'}"
        )

    best = max(ok, key=lambda t: t.target)
    aborted = False
    abort_reason = ""
    if len(ok) < cfg.n_iters and len(trials) < cfg.n_iters:
        aborted = True
        consecutive = _trailing_failures(trials)
        abort_reason = (
            f"连续 {consecutive} 次训练异常，达到熔断阈值 "
            f"{cfg.max_consecutive_failures}，提前中止。"
            f"返回熔断前已成功的最好结果（共 {len(trials)} 次试验 / "
            f"计划 {cfg.n_iters} 次）。"
        )

    return TuningResult(
        best_params=dict(best.params),
        best_target=float(best.target),
        trials=trials,
        best_metrics=dict(best.metrics),
        aborted=aborted,
        abort_reason=abort_reason,
        engine=cfg.engine,
        seed=cfg.seed,
        n_iters_requested=cfg.n_iters,
    )


def _trailing_failures(trials: Sequence[TuningTrial]) -> int:
    n = 0
    for t in reversed(trials):
        if t.status == "failed":
            n += 1
        else:
            break
    return n


def _assemble(full: Mapping[str, Any], sampled: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(full), **dict(sampled)}


def _run_one(
    params: Mapping[str, Any],
    index: int,
    cfg: TuningConfig,
    trial_fn: TrialFn,
) -> TuningTrial:
    started = time.perf_counter()
    try:
        metrics, best_iter = trial_fn(dict(params), index)
        metrics = apply_gap_threshold(metrics, cfg.gap_threshold)
        target = _objective_value(metrics, cfg.weights, gap_threshold=cfg.gap_threshold)
        if np.isnan(target):
            raise ModelError(
                f"目标函数无法计算，指标中 KS/AUC 缺失：{dict(metrics)}。"
                f"通常是验证集里好/坏样本未同时存在。"
            )
        return TuningTrial(
            index=index,
            params=dict(params),
            status="ok",
            target=float(target),
            metrics=dict(metrics),
            elapsed_sec=time.perf_counter() - started,
        )
    except ConfigError:
        # ★ 配置错误必须往上抛，不能当成「这组参数效果差」
        #   典型场景：固定参数与 LightGBM 冲突、搜索空间声明自相矛盾。
        #   若被捕获成 failed，会让优化器误以为那一片区域不好，
        #   最终熔断并返回「一个错误的参数」—— 报错信息完全丢失。
        raise
    except (ModelError, ValueError, TypeError, RuntimeError) as exc:
        # ★ 失败也要进轨迹：不记录 = 优化器以为那组参数「效果差」，
        #   会被系统性地带偏搜索方向
        return TuningTrial(
            index=index,
            params=dict(params),
            status="failed",
            target=float("nan"),
            error=f"{type(exc).__name__}: {exc}"[:300],
            elapsed_sec=time.perf_counter() - started,
        )


def _search_hyperopt(
    cfg: TuningConfig, trial_fn: TrialFn
) -> list[TuningTrial]:
    """``hyperopt`` 引擎（Tree of Parzen Estimators，贝叶斯优化）。

    ★ 本模块唯一的调参引擎
    ----------------------
    TPE 会利用历史试验结果，把采样集中在表现好的区域 ——
    同样预算下比等概率随机采样更早找到好参数。

    ★ 必须设 ``rstate`` 与 LightGBM ``random_state``
    ------------------------------------------------
    hyperopt 的 TPE 内部有随机采样（初始随机探索 + 候选采样），
    不固定 ``rstate`` 则每次搜索出的最优参数都不同。
    """
    try:
        from hyperopt import Trials, fmin, tpe
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise ConfigError(
            "未安装 hyperopt。请执行 pip install hyperopt"
            "（hyperopt 是本流程唯一的调参引擎）。"
        ) from exc

    objective = _HyperoptObjective(cfg, trial_fn)
    trials_obj = Trials()
    space = _to_hyperopt_space(cfg.space)

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=cfg.n_iters,
        trials=trials_obj,
        rstate=np.random.default_rng(cfg.seed),
        show_progressbar=False,
        verbose=False,
    )
    return objective.records


def _to_hyperopt_space(space: SearchSpace) -> dict[str, Any]:
    """把声明式空间翻译成 hyperopt 的 ``hp.*``。

    ★★ 这里有四处**与直觉相反**的语义，绝不能照字面写
    ------------------------------------------------------
    逐条查实于 ``hyperopt`` 源码（``pyll.stochastic``）并用数百次采样验证：

    1. ``hp.loguniform(low, high)``  实际是  ``exp(uniform(low, high))``。

       ``low`` / ``high`` 是**指数空间的边界**，不是取值本身的边界。
       照字面写 ``hp.loguniform('lr', 0.01, 0.2)`` 得到的是
       ``learning_rate ≈ exp(0.1) ≈ 1.1`` —— 而 LightGBM 的学习率
       有效区间是 0~1，1.1 的学习率会让模型**几乎不分裂**。
       ★ 更危险的是它**不报错**：整轮调参会正常跑完、轨迹正常落盘、
       最终产出一个「AUC 接近 0.5 的垃圾模型」，而所有报表都显示「调参成功」。

       正确写法是传**取过对数的边界**：``hp.loguniform(name, log(low), log(high))``。

    2. ``hp.choice(name, options)`` 返回**索引**，不是选项值本身。

       ``options=[2, 3, 4, 5]`` 会返回 ``0/1/2/3`` ——
       于是 ``max_depth=0``（LightGBM 视为不限深度）、``num_leaves`` 可能变成
       索引 0 也就是列表首元素。**同样不报错。**

       官方推荐用 ``pyll.rec_eval`` 还原，但它必须在目标函数**内部**调用，
       且对已求值的整数会报 ``deepcopy_inputs should be bool``（实测）。
       本实现改用「采样整数索引 + 在目标函数里显式映射」。

    3. ``hp.randint(label, low, high)`` 是**左闭右开** —— ``[5, 20)`` 取不到 20。

       本模块对整数参数的声明语义是**闭区间**。
       两者不一致 → 「声明范围」的实际含义会变。
       统一为闭区间：上界 +1。

    4. ★★ ``hp.<dist>(label, ...)`` 的 ``label`` 参数**在 dict 形态下被忽略** ——
       返回给目标函数的键名用的是**外层 Python dict 的键**，不是 ``label``。

       所以「给 label 加后缀来区分索引列」这种写法**根本不生效**：

           {"max_depth": hp.randint("max_depth__idx", 0, 3)}
           → 目标函数收到 {"max_depth": 0}      # __idx 消失，键名还是 max_depth

       实测确认。正确做法是把索引列放进**独立子 dict**，用子 dict 的键名区分。

    这四条合在一起构成一个非常隐蔽的失效：**配置与代码都「看起来对」，
    跑起来也全绿，但搜出来的参数完全不是声明的那个区间**。
    唯一可靠的防线是**对转换结果做取值域断言**（见 :func:`_assert_in_domain`），
    而且必须在真实调参路径上跑，而不是只测转换函数。
    """
    from hyperopt import hp

    out: dict[str, Any] = {}
    choice_idx: dict[str, Any] = {}
    for name, spec in space.params.items():
        kind = spec["type"]
        if kind == "choice":
            # ★ 放进独立子 dict：hp.* 忽略 label，键名只能靠 dict 层级区分
            # ★★ 单选项 choice 必须**退化为常量**、不能交给 hyperopt：
            #    hp.randint(name, 0, len-1) 在 len==1 时 high==0，
            #    hyperopt 内部直接抛 `ValueError: high <= 0`
            #    ⇒ 整个调参步骤失败。
            #    （random 引擎按值抽取，从不暴露这条路径 ⇒ 这是下线 random
            #     之后由单测抓出来的**真实缺陷**，不是测试写错。）
            #    语义上等价：唯一选项就是唯一取值，没有任何可采样空间。
            if len(spec["options"]) == 1:
                out[name] = spec["options"][0]
                continue
            choice_idx[name] = hp.randint(name, 0, len(spec["options"]) - 1)
            continue

        # ★ 退化空间（low == high）：直接固定，不交给 hyperopt
        #   hp.quniform(a, a, step) 在等值边界上行为不确定，
        #   交给它采样等于把一个确定的常量交给随机数。
        if float(spec["high"]) == float(spec["low"]):
            out[name] = spec["low"]
            continue

        if kind == "uniform":
            out[name] = hp.uniform(name, float(spec["low"]), float(spec["high"]))
        elif kind == "quniform":
            out[name] = hp.quniform(
                name, float(spec["low"]), float(spec["high"]), float(spec["step"])
            )
        elif kind == "loguniform":
            out[name] = hp.loguniform(
                name, float(np.log(float(spec["low"]))), float(np.log(float(spec["high"])))
            )
        elif kind == "qloguniform":
            out[name] = hp.qloguniform(
                name,
                float(np.log(float(spec["low"]))),
                float(np.log(float(spec["high"]))),
                float(spec["step"]),
            )
        else:
            out[name] = hp.randint(name, int(spec["low"]), int(spec["high"]) + 1)

    if choice_idx:
        out[_CHOICE_BUCKET] = choice_idx
    return out


_CHOICE_BUCKET = "__choices__"
"""choice 索引列在 hyperopt 空间里的独立子 dict 键名。

见 :func:`_to_hyperopt_space` 第4 条说明：``hp.*`` 的 label 参数在 dict 形态下
被忽略，只能靠 dict 层级区分。
"""


def _restore_choice(raw: Mapping[str, Any], space: SearchSpace) -> dict[str, Any]:
    """把子 dict 里的索引还原成声明的真实取值。

    ★ 顺带把 numpy 标量转成 Python 原生类型
    ---------------------------------------
    hyperopt 返回的是 ``np.int64`` / ``np.float64``。它们在算术上与原生类型
    等价，但**序列化行为不同**：``json.dumps(np.int64(145))`` 直接抛
    ``TypeError: Object of type int64 is not JSON serializable`` ——
    而 Tool 返回体、参数文档、交付物都要过 JSON。

    更隐蔽的是 ``np.float64`` 与 float 混算后结果类型会变，
    导致「同一个参数在训练时是 np.float64、在落盘时是 float」这种
    口径不一致。
    """
    out: dict[str, Any] = {
        k: _native(v) for k, v in raw.items() if k != _CHOICE_BUCKET
    }
    bucket = raw.get(_CHOICE_BUCKET) or {}
    for name, idx in bucket.items():
        out[name] = _native(space.params[name]["options"][int(idx)])

    # ★ 整数参数统一转 int
    #   `hp.quniform` 按连续分布采样 → 浮点。`min_data_in_leaf` 这类整数参数
    #   拿到浮点会被 LightGBM 直接拒绝：
    #     Parameter min_data_in_leaf should be of type int, got "190.0"
    # ★ 必须在这里做，而不是只在采样器里做
    #   hyperopt 路径（采样索引还原 + 退化常量）都经过 `_restore_choice` ——
    #   在这里修一次，所有取值来源都覆盖。
    # 只在「声明为整数参数」时转，避免把 `learning_rate` 这类
    # 浮点参数误转成 int（那会静默丢精度，更难发现）。
    for name in list(out):
        if name in _INT_ONLY_PARAMS and isinstance(out[name], float):
            out[name] = int(out[name])
    return out


def _native(value: Any) -> Any:
    """numpy 标量 → Python 原生类型。非 numpy 值原样返回。"""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _assert_in_domain(
    sampled: Mapping[str, Any], space: SearchSpace, *, context: str
) -> None:
    """断言实际取值落在声明范围内。**这是唯一能抓住上述语义错位的手段。**

    直觉与文档都说「空间就是声明的那个区间」，但真跑起来才发现不是。
    没有断言时，这类错误一路走到训练阶段才暴露，而且表现为「模型效果差」
    而非「参数越界」—— 排查成本极高。

    ★★ **类型也必须校验**
    ----------------------
    原本只校验取值域。实测踩到：``min_data_in_leaf`` 声明为
    ``quniform``（step=10），采样结果是 **190.0**（浮点）——
    取值域 [20, 300] 完全合法，断言通过；
    但 LightGBM 要求该参数为 **int**，直接抛
    ``Parameter min_data_in_leaf should be of type int, got "190.0"``。

    ★ 判据：哪些参数必须是 int
    ----------------------
    由 LightGBM 的参数类型决定。本实现用一张**显式白名单**判定，
    而不是靠参数名猜测 —— 猜错的后果是「整数参数被当浮点传下去」
    或「浮点参数被误转 int」。
    """
    for name, value in sampled.items():
        spec = space.params.get(name)
        if spec is None:
            continue
        if spec["type"] == "choice":
            if value not in spec["options"]:
                raise ConfigError(
                    f"{context}：参数 {name} 取值 {value!r} 不在声明的选项 "
                    f"{spec['options']} 内。\n"
                    f"★ 这几乎必然是索引未还原导致 —— hyperopt 的 choice 类分布"
                    f"返回的是索引，必须经 `_restore_choice` 映射回真实取值。"
                )
            continue

        # ★★ 类型校验：整数参数不能拿到浮点值
        if name in _INT_ONLY_PARAMS and isinstance(value, float) and not value.is_integer():
            raise ConfigError(
                f"{context}：参数 {name} 必须是整数，但采样得到 {value!r}。\n"
                f"★ 声明类型为 {spec['type']} —— 它按连续分布采样，产出浮点。\n"
                f"  整数参数请改用 {{\"type\": \"int\", \"low\": …, \"high\": …}}，"
                f"或用 quniform 并声明 step（此时服务会按 step 取整）。"
            )

        try:
            f = float(value)
        except (TypeError, ValueError):
            raise ConfigError(
                f"{context}：参数 {name} 取值 {value!r} 不是数值，"
                f"但声明类型为 {spec['type']}（应为连续取值）。"
                f"若是 choice 声明，请确认 options 里的元素本身是数值。"
            ) from None
        lo, hi = float(spec["low"]), float(spec["high"])
        if not (lo <= f <= hi):
            raise ConfigError(
                f"{context}：参数 {name} 取值 {f} 越出声明范围 [{lo}, {hi}]。\n"
                f"★ 若声明类型是 loguniform / qloguniform，几乎必然是 hyperopt 的"
                f" `loguniform` 语义为 `exp(uniform(low, high))` 所致 —— "
                f"必须传 log 后的边界，不能传取值本身的边界。"
            )


#: 必须是**整数**的 LightGBM 参数（采样出浮点会被 LightGBM 拒绝）。
#:
#: ★ 为什么用显式白名单而不是「看参数名猜」
#: --------------------------------------
#: 猜错的后果双向都坏：整数参数被当浮点传下去 → LightGBM 直接报错；
#: 浮点参数被误转 int → 精度丢失但不报错，**更难发现**。
#: 所以这张表必须显式、可审阅、可随 LightGBM 版本更新。
_INT_ONLY_PARAMS: frozenset[str] = frozenset({
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "min_child_samples",
    "min_child_weight",
    "bagging_freq",
    "feature_fraction_seed",
    "bagging_seed",
    "data_random_seed",
    "num_iterations",
    "max_bin",
    "subsample_freq",
    "cat_smooth",
    "cat_l2",
    "path_smooth",
    "seed",
})


class _HyperoptObjective:
    """把 hyperopt 的「最小化 raw_loss」适配成我们的「最大化目标值」。

    同时承担三件事：
    1. 取负号 —— hyperopt 只会最小化；
    2. 把每次调用的参数与结果**记录下来** —— hyperopt 内部的 ``Trials``
       不便于直接做报表（它的 loss 是取负后的值，读报表的人要自己再取负）；
    3. 熔断 —— hyperopt 没有「跑够就停」的概念，只能靠抛异常中断，
       而抛异常会被 hyperopt 当作「这次 loss 无穷大」继续跑，所以必须
       **直接把状态记录下来，由外层在超限后停止**。
    """

    def __init__(self, cfg: TuningConfig, trial_fn: TrialFn) -> None:
        self._cfg = cfg
        self._trial_fn = trial_fn
        self.records: list[TuningTrial] = []
        self._consecutive = 0
        self._halted = False

    def __call__(self, raw: Mapping[str, Any]) -> float:
        if self._halted:
            # 已熔断：返回一个恒定的大值，不再真正训练
            return 1e6

        restored = _restore_choice(raw, self._cfg.space)
        _assert_in_domain(
            restored, self._cfg.space, context=f"第 {len(self.records) + 1} 次采样"
        )
        params = _assemble(self._cfg.fixed_params, restored)
        index = len(self.records) + 1
        trial = _run_one(params, index, self._cfg, self._trial_fn)
        self.records.append(trial)

        if trial.status == "failed":
            self._consecutive += 1
            if self._consecutive >= self._cfg.max_consecutive_failures:
                self._halted = True
        else:
            self._consecutive = 0

        return -trial.target if not np.isnan(trial.target) else 1e6
