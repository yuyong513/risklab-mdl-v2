"""``risklab_mdl.platform`` —— 跨平台基础设施层。

★ 为什么单独一层
----------------
早期版本把「原子写 + 文件锁 + 路径处理」散在 ``utils.py`` / ``locking.py``，
且 ``locking.py`` 直接 ``import fcntl`` ⇒ Windows 上导入即崩。本迭代版把
所有与操作系统相关的写盘细节收敛到 ``platform.fs``，其余模块一律只调
这里的函数，从而保证「同一份代码在 Windows / macOS / Linux 行为一致」。

本层**不 import fcntl**（POSIX-only），锁使用 ``filelock`` 包。
"""

from .fs import (
    LockTimeout,
    PathNotReadable,
    PathNotWritable,
    PlatformError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    check_writable,
    detect_encoding,
    ensure_dir,
    ensure_writable_dir,
    file_lock,
    is_writable,
    normalize_path,
    read_json,
    read_text_auto,
    require_readable,
)

__all__ = [
    "LockTimeout",
    "PathNotReadable",
    "PathNotWritable",
    "PlatformError",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "check_writable",
    "detect_encoding",
    "ensure_dir",
    "ensure_writable_dir",
    "file_lock",
    "is_writable",
    "normalize_path",
    "read_json",
    "read_text_auto",
    "require_readable",
]
