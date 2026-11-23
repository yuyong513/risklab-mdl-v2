"""自造测试数据生成器（T2 core 单测与 T3/T4/T5 复用）。

★ 为什么自造而不用 ``sample_2000.csv``
------------------------------------
1. 旧项目样例数据**不在本仓库**（按 Q4 只读引用），单测不能依赖外部路径；
2. 该样例极不平衡（35/2000 坏样本）⇒ val 里只有约 8 个坏样本，
   任何「KS/PSI 阈值」类断言都会因抽样噪声随机红绿 —— 这是**测试自身的缺陷**，
   不是被测代码的问题；
3. core 层要验的是**口径**（分箱/WOE/指标/划分），与具体业务数据无关。

⇒ 本模块用固定 seed 生成三份小数据，保证：
* 平衡二分类（好/坏各半）—— 任何分类器都能算出非退化指标；
* 三分类（三类各 1/3）—— 多分类 OVR 每类都有足够正负例；
* 含缺失 / 文本 / 常量 / 高相关列 —— 覆盖预处理与筛选分支。

用法：:

    from fixtures.make_fixtures import binary_frame, multiclass_frame, write_all
    df = binary_frame()               # 内存模式（单测首选）
    paths = write_all(tmp_dir)        # 落盘模式（T3/T4/T5 端到端用）

直接运行本文件会把三份 CSV 写到 ``tests/fixtures/data/``（便于人工查看）。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "BINARY_CSV",
    "MISSING_TEXT_CSV",
    "MISSING_TEXT_MONTH_CSV",
    "MULTICLASS_CSV",
    "NO_TIME_CSV",
    "binary_frame",
    "missing_text_frame",
    "missing_text_month_frame",
    "multiclass_frame",
    "no_time_frame",
    "prepare_project",
    "set_target",
    "write_all",
    "write_encoded",
]

BINARY_CSV = "binary_small.csv"
MULTICLASS_CSV = "multiclass_small.csv"
MISSING_TEXT_CSV = "missing_text_small.csv"
MISSING_TEXT_MONTH_CSV = "missing_text_month_small.csv"
NO_TIME_CSV = "no_time_small.csv"

_MONTHS = ("2026-01", "2026-02", "2026-03")


def binary_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """平衡二分类小数据（好/坏各半）。

    列：

    ============  ========================================================
    列             含义
    ============  ========================================================
    ``id``        唯一标识（角色列，不应入模）
    ``y``         标签 0/1，**严格平衡**
    ``month``     3 个月（PSI / 分月评估用）
    ``split``     train/val/test 标签（旧口径的切分列）
    ``f_signal``  与 y 同向的强信号（AUC 约 0.8）
    ``f_weak``    弱信号
    ``f_noise``   纯噪声
    ``f_dup``     ``f_signal`` 的高相关副本（相关性筛选用，r≈0.95）
    ``f_missing`` 20% 缺失的数值列
    ``f_text``    4 档文本列（含 ``None``）+ 与 y 弱相关
    ``f_const``   常量列（同值率筛选用）
    ============  ========================================================
    """
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2) + ([1] if n % 2 else []), dtype=int)
    rng.shuffle(y)

    signal = rng.normal(size=n) + 1.5 * y
    weak = rng.normal(size=n) + 0.5 * y
    noise = rng.normal(size=n)
    dup = signal + rng.normal(scale=0.15, size=n)
    missing = rng.normal(size=n)
    missing[rng.random(n) < 0.2] = np.nan

    text_levels = np.array(["A", "B", "C", "D"], dtype=object)
    text = text_levels[rng.integers(0, 4, size=n)]
    text[rng.random(n) < 0.05] = None

    split = np.array(["train"] * int(n * 0.6) + ["val"] * int(n * 0.2)
                     + ["test"] * (n - int(n * 0.6) - int(n * 0.2)), dtype=object)
    rng.shuffle(split)

    return pd.DataFrame(
        {
            "id": [f"U{i:05d}" for i in range(n)],
            "y": y,
            "month": np.array(
                [_MONTHS[i % len(_MONTHS)] for i in range(n)], dtype=object
            ),
            "split": split,
            "f_signal": signal,
            "f_weak": weak,
            "f_noise": noise,
            "f_dup": dup,
            "f_missing": missing,
            "f_text": text,
            "f_const": np.ones(n),
        }
    )


def multiclass_frame(n: int = 360, seed: int = 11) -> pd.DataFrame:
    """平衡三分类小数据（0/1/2 各 1/3）。

    ``f1`` 区分 0 vs 其余；``f2`` 区分 1 vs 其余；``f3`` 区分 2 vs 其余；
    ``f_text`` 是含缺失的文本列；``month`` 三个月。
    """
    rng = np.random.default_rng(seed)
    labels = np.array([0, 1, 2] * (n // 3) + list(range(n % 3)), dtype=int)
    rng.shuffle(labels)

    f1 = rng.normal(size=n) + np.where(labels == 0, 2.0, 0.0)
    f2 = rng.normal(size=n) + np.where(labels == 1, 2.0, 0.0)
    f3 = rng.normal(size=n) + np.where(labels == 2, 2.0, 0.0)
    noise = rng.normal(size=n)
    text = np.array(["low", "mid", "high"], dtype=object)[labels]
    text[rng.random(n) < 0.05] = None
    return pd.DataFrame(
        {
            "id": [f"M{i:05d}" for i in range(n)],
            "y": labels,
            "month": np.array([_MONTHS[i % 3] for i in range(n)], dtype=object),
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f_noise": noise,
            "f_text": text,
        }
    )


def missing_text_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """缺失/文本/哨兵值混杂小数据（预处理分支用）。

    * ``amount``：含 30% 缺失 + 少量 99999 哨兵值；
    * ``channel``：文本列，含 ``None`` 与 ``""``；
    * ``score``：右偏 + 极端值（盖帽用）；
    * ``const``：常量列；
    * ``y``：平衡 0/1。
    """
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2) + ([1] if n % 2 else []), dtype=int)
    rng.shuffle(y)

    amount = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    pad = rng.random(n)
    amount[pad < 0.30] = np.nan
    amount[(pad >= 0.30) & (pad < 0.35)] = 99999.0

    channel = np.array(["app", "web", "store"], dtype=object)[rng.integers(0, 3, n)]
    channel[rng.random(n) < 0.10] = None
    channel[rng.random(n) < 0.05] = ""

    score = rng.normal(size=n)
    score[rng.random(n) < 0.02] = 50.0  # 极端值

    return pd.DataFrame(
        {
            "id": [f"T{i:05d}" for i in range(n)],
            "y": y,
            "amount": amount,
            "channel": channel,
            "score": score,
            "const": np.ones(n),
        }
    )


def missing_text_month_frame(n: int = 180, seed: int = 13) -> pd.DataFrame:
    """缺失 + 文本 + **时间列 / split 列** 的小数据（T4 预处理与初筛端到端用）。

    与 :func:`missing_text_frame` 的区别：带 ``month``（PSI / 时间口径）与 ``split``
    （train/val/test 切分列），是「有划分信息」的那一版；``missing_text_frame`` 是
    「无时间列/无划分」的那一版 —— 两版一起覆盖「时间列/无时间列」两条分支。

    列：

    ============  ========================================================
    列             含义
    ============  ========================================================
    ``id``        唯一标识
    ``y``         标签 0/1（严格平衡）
    ``month``     3 个月
    ``split``     train/val/test
    ``amount``    30% 缺失 + 少量 99999 哨兵值（缺失 / 哨兵冲突）
    ``channel``   4 档文本（含 None 与 ""，编码分支）
    ``score``     右偏 + 极端值 50（盖帽分支）
    ``const``     常量列（同值列）
    ``note``      高基数文本（60 档，高基数类别建议）
    ============  ========================================================
    """
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2) + ([1] if n % 2 else []), dtype=int)
    rng.shuffle(y)

    amount = rng.lognormal(mean=6.0, sigma=1.0, size=n)
    pad = rng.random(n)
    amount[pad < 0.30] = np.nan
    amount[(pad >= 0.30) & (pad < 0.35)] = 99999.0

    channel = np.array(["app", "web", "store", "partner"], dtype=object)[
        rng.integers(0, 4, n)
    ]
    channel[rng.random(n) < 0.10] = None
    channel[rng.random(n) < 0.05] = ""

    score = rng.normal(size=n)
    score[rng.random(n) < 0.02] = 50.0

    note = np.array([f"n{i:02d}" for i in range(60)], dtype=object)[
        rng.integers(0, 60, n)
    ]
    split = np.array(
        ["train"] * int(n * 0.6)
        + ["val"] * int(n * 0.2)
        + ["test"] * (n - int(n * 0.6) - int(n * 0.2)),
        dtype=object,
    )
    rng.shuffle(split)

    return pd.DataFrame(
        {
            "id": [f"P{i:05d}" for i in range(n)],
            "y": y,
            "month": np.array([_MONTHS[i % 3] for i in range(n)], dtype=object),
            "split": split,
            "amount": amount,
            "channel": channel,
            "score": score,
            "const": np.ones(n),
            "note": note,
        }
    )


def no_time_frame(n: int = 200, seed: int = 5) -> pd.DataFrame:
    """**没有时间列**的二分类小数据（T3 验收 ④：跳月份图/交叉表且告警）。

    同时保留文本列（``seg``）与高缺失列（``f_sparse``），方便一次覆盖
    「无时间列 + 文本跳过 + 缺失标注」三种分支。
    """
    rng = np.random.default_rng(seed)
    y = np.array([1, 0] * (n // 2) + ([1] if n % 2 else []), dtype=int)
    rng.shuffle(y)
    signal = rng.normal(size=n) + 1.2 * y
    sparse = rng.normal(size=n)
    sparse[rng.random(n) < 0.55] = np.nan
    seg = np.array(["seg_a", "seg_b", "seg_c"], dtype=object)[rng.integers(0, 3, size=n)]
    return pd.DataFrame(
        {
            "apply_no": [f"N{i:05d}" for i in range(n)],
            "y": y,
            "f_signal": signal,
            "f_sparse": sparse,
            "seg": seg,
        }
    )


def write_encoded(
    path: str | Path, frame: pd.DataFrame, *, encoding: str = "utf-8"
) -> Path:
    """按指定编码落一份 CSV（编码探测测试用：utf-8 / gb18030 / utf-8-sig）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding=encoding)
    return target


def set_target(
    project_dir: str | Path,
    *,
    column: str = "y",
    task: str = "binary",
    labels: Dict[str, str] | None = None,
) -> None:
    """把项目目标字段写进 manifest（T5 模型测试的项目初始化后置步骤）。"""
    from risklab_mdl.artifacts.manifest import update_manifest

    def _mutate(doc: Dict[str, object]) -> None:
        project = doc["project"]  # type: ignore[index]
        project["target"] = {"column": column, "task": task, "labels": labels}

    update_manifest(str(project_dir), _mutate)


def prepare_project(
    tmp_dir: str | Path,
    frame: pd.DataFrame,
    *,
    name: str = "binary_small",
    target_column: str = "y",
    task: str = "binary",
    labels: Dict[str, str] | None = None,
) -> tuple[Path, str]:
    """写 CSV → ``project_init`` → 写入目标字段，返回 ``(csv_path, project_dir)``。

    ★ 为什么测试也要走真实工具而不是手搭目录
    --------------------------------------
    ``project_init`` 负责建齐 8 个固定子目录并写 manifest；手搭目录的测试会在
    「子目录少一个」「manifest 缺 schema」这类真实故障上**假绿**。
    """
    from risklab_mdl.tools.dataset import project_init

    out_dir = Path(tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    payload = project_init(str(csv_path))
    project_dir = payload["data"]["project_dir"]
    set_target(project_dir, column=target_column, task=task, labels=labels)
    return csv_path, project_dir


def write_all(directory: str | Path) -> Dict[str, Path]:
    """把自造数据写成 CSV（UTF-8 无 BOM），返回 ``{名字: 路径}``。

    ★ UTF-8 无 BOM 是有意为之：T3 的编码探测需要覆盖「无 BOM」这条最常见路径，
      带 BOM 的样本由 T3 自己的测试构造。
    """
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "binary": out_dir / BINARY_CSV,
        "multiclass": out_dir / MULTICLASS_CSV,
        "missing_text": out_dir / MISSING_TEXT_CSV,
        "missing_text_month": out_dir / MISSING_TEXT_MONTH_CSV,
        "no_time": out_dir / NO_TIME_CSV,
    }
    binary_frame().to_csv(paths["binary"], index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    multiclass_frame().to_csv(paths["multiclass"], index=False, encoding="utf-8")
    missing_text_frame().to_csv(paths["missing_text"], index=False, encoding="utf-8")
    missing_text_month_frame().to_csv(
        paths["missing_text_month"], index=False, encoding="utf-8"
    )
    no_time_frame().to_csv(paths["no_time"], index=False, encoding="utf-8")
    return paths


if __name__ == "__main__":  # pragma: no cover - 手工生成用
    target = Path(__file__).resolve().parent / "data"
    for name, path in write_all(target).items():
        print(f"{name}: {path}")
