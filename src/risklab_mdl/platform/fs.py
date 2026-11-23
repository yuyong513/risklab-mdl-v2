"""跨平台文件基础设施：原子写、文件锁、路径校验、编码安全读写。

★ 设计三条铁律
--------------
1. **不 import fcntl**（POSIX-only）。锁用 ``filelock``，它在 Windows 上走
   ``msvcrt.locking``、在 POSIX 上走 ``fcntl.flock``，对外接口一致。
2. **不静默降级**。拿不到锁就抛 :class:`LockTimeout`；目录不可写就抛
   :class:`PathNotWritable`；JSON 损坏就抛 :class:`PlatformError`。
   早期版本的教训是「失败不可见」——写入被吞掉比报错更难查。
3. **原子写的唯一实现**：同目录临时文件 → ``flush`` + ``fsync`` →
   ``os.replace``（Windows / POSIX 语义一致，均为原子替换）。

错误码约定（供 MCP 三态协议 ``error.code`` 使用）：
``path_not_writable`` / ``path_not_readable`` / ``lock_timeout`` /
``json_decode_error`` / ``encoding_undecodable`` / ``platform_error``。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

PathLike = Union[str, os.PathLike]

#: 有无 BOM 的 UTF-8 探测用这个前缀判定。
_UTF8_BOM = b"\xef\xbb\xbf"

#: UTF-16 BOM（LE = ``FF FE``，BE = ``FE FF``）。Excel「另存为 Unicode 文本」产出的
#: CSV 就是带这两个 BOM 的 UTF-16：不识别时会落到 gb18030 兜底，读出满屏乱码。
_UTF16_LE_BOM = b"\xff\xfe"
_UTF16_BE_BOM = b"\xfe\xff"

#: 探测失败时按此顺序尝试（gb18030 覆盖 GBK/GB2312，是中文环境的最终兜底）。
_DETECT_CHAIN = ("utf-8", "gb18030")

try:  # pragma: no cover - 依赖缺失在 doctor 里单独报
    from filelock import FileLock
    from filelock import Timeout as _FileLockTimeout

    _HAS_FILELOCK = True
except Exception:  # noqa: BLE001
    FileLock = None  # type: ignore[assignment]
    _FileLockTimeout = ()  # type: ignore[assignment]
    _HAS_FILELOCK = False


class PlatformError(RuntimeError):
    """平台层错误的基类。``code`` 直接进 MCP 错误载荷。"""

    code = "platform_error"

    def __init__(self, message: str, *, hint: str = "", code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        if code:
            self.code = code

    def to_payload(self) -> Dict[str, str]:
        """转成 MCP 三态协议里的 ``error`` 载荷。"""
        return {"code": self.code, "message": self.message, "hint": self.hint}


class PathNotReadable(PlatformError):
    """路径不存在或不可读。"""

    code = "path_not_readable"


class PathNotWritable(PlatformError):
    """路径不可写（含父目录不可写、只读挂载、权限不足）。"""

    code = "path_not_writable"


class LockTimeout(PlatformError):
    """在超时时间内没能拿到文件锁。"""

    code = "lock_timeout"


class EncodingUndecodable(PlatformError):
    """按探测到的编码无法解码（多为二进制文件、加密文件或半截文件）。"""

    code = "encoding_undecodable"


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #


def normalize_path(path: PathLike) -> Path:
    """展开 ``~``、转绝对路径、规范化分隔符（不改大小写、不要求存在）。

    Windows 上同时接受 ``/`` 与 ``\\``；``resolve()`` 会消掉 ``..`` 段，
    避免 ``mdl-x/../../etc`` 这类路径穿越。
    """
    if isinstance(path, os.PathLike):
        raw = os.fspath(path)
    else:
        raw = str(path)
    if not raw:
        raise PlatformError("路径为空", code="path_invalid", hint="请传入非空的文件或目录路径")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def ensure_dir(path: PathLike) -> Path:
    """确保目录存在（递归创建），返回规范化路径。"""
    p = normalize_path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PathNotWritable(
            f"无法创建目录：{p}（{type(exc).__name__}: {exc}）",
            hint="检查父目录权限；CSV 所在目录不可写时请换可写位置或调整权限",
        ) from exc
    if not p.is_dir():
        raise PathNotWritable(f"路径已存在但不是目录：{p}", hint="删除同名文件或换一个目录名")
    return p


def is_writable(target: PathLike) -> bool:
    """实测一个路径是否可写（写探针文件后立即删除，不留痕）。

    对已存在的文件判断能否追加/覆盖；对不存在的路径判断**最近的已存在父目录**。
    """
    try:
        p = normalize_path(target)
    except PlatformError:
        return False
    if p.exists() and p.is_file():
        try:
            with p.open("ab"):
                return True
        except OSError:
            return False
    probe = p if p.exists() else p.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        return False
    name = ""
    try:
        fd, name = tempfile.mkstemp(dir=str(probe), prefix=".risklab-writecheck-", suffix=".tmp")
        os.close(fd)
        os.unlink(name)
        return True
    except OSError:
        if name:
            try:
                os.unlink(name)
            except OSError:
                pass
        return False


def check_writable(target: PathLike, *, label: str = "目录") -> Path:
    """可写性校验；不可写抛 :class:`PathNotWritable`（不静默换位置）。"""
    p = normalize_path(target)
    if not is_writable(p):
        raise PathNotWritable(
            f"{label}不可写：{p}",
            hint=(
                "该位置所在磁盘/目录没有写权限（只读挂载、受保护目录或权限不足）。"
                "请把 CSV 放到可写目录，或修正目录权限后重试；"
                "本工具不会静默改写到其他位置。"
            ),
        )
    return p


def ensure_writable_dir(path: PathLike, *, label: str = "目录") -> Path:
    """确保目录存在**且**可写。"""
    p = ensure_dir(path)
    return check_writable(p, label=label)


def require_readable(path: PathLike, *, label: str = "文件") -> Path:
    """存在性 + 可读性校验。"""
    p = normalize_path(path)
    if not p.exists():
        raise PathNotReadable(
            f"{label}不存在：{p}",
            hint="确认路径拼写；Windows 路径请用原始字符串或双反斜杠",
        )
    if p.is_dir():
        raise PathNotReadable(f"{label}是目录而不是文件：{p}", hint="请传入具体文件名")
    try:
        with p.open("rb") as fh:
            fh.read(1)
    except OSError as exc:
        raise PathNotReadable(
            f"{label}不可读：{p}（{type(exc).__name__}: {exc}）",
            hint="检查文件权限，或该文件是否被其他程序独占打开",
        ) from exc
    return p


# --------------------------------------------------------------------------- #
# 编码探测与文本读取
# --------------------------------------------------------------------------- #


def detect_encoding(path: PathLike, *, sample_size: int = 1 << 20) -> str:
    """探测编码：按 BOM 优先，其次严格试 ``utf-8``，最后退 ``gb18030``。

    - UTF-8 BOM ⇒ ``utf-8-sig``；UTF-16 LE/BE BOM（``FF FE`` / ``FE FF``）⇒ ``utf-16``；
    - 无 BOM：先试 ``utf-8``，再退 ``gb18030``（覆盖 GBK/GB2312，是中文环境的兜底）。

    探测用**严格**解码：能严格解开即采用。``gb18030`` 也解不开时本函数仍返回
    ``gb18030``（不在此处抛），由 :func:`read_text_auto` 在真正读取时转成
    ``encoding_undecodable`` 的 typed error —— 这样「只是探测」的调用方不会因为
    二进制文件而崩，而真正读文件的一方一定会看到明确错误。
    """
    p = require_readable(path)
    with p.open("rb") as fh:
        sample = fh.read(sample_size)
    if sample.startswith(_UTF8_BOM):
        return "utf-8-sig"
    if sample.startswith(_UTF16_LE_BOM) or sample.startswith(_UTF16_BE_BOM):
        return "utf-16"
    for enc in _DETECT_CHAIN:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "gb18030"


def read_text_auto(path: PathLike) -> str:
    """按探测到的编码读全文（不丢 BOM 语义）。

    解码失败（二进制/加密/半截文件）⇒ :class:`EncodingUndecodable`
    （``code=encoding_undecodable``），不把裸 ``UnicodeDecodeError`` 抛给调用方。
    """
    p = require_readable(path)
    enc = detect_encoding(p)
    try:
        with p.open("r", encoding=enc, newline="") as fh:
            return fh.read()
    except UnicodeDecodeError as exc:
        raise EncodingUndecodable(
            f"文件无法按编码 {enc} 解码：{p}（字节位置 {exc.start}：{exc.reason}）",
            hint=(
                "该文件可能不是文本（二进制/加密/压缩），或编码不在支持范围。"
                "请用 Excel/编辑器另存为 UTF-8（或 UTF-16、GB18030）后重试；"
                "不要用二进制文件冒充 CSV。"
            ),
        ) from exc


# --------------------------------------------------------------------------- #
# 原子写
# --------------------------------------------------------------------------- #


def atomic_write_bytes(path: PathLike, data: bytes) -> Path:
    """原子写字节：同目录临时文件 → fsync → ``os.replace``。"""
    p = normalize_path(path)
    parent = ensure_writable_dir(p.parent, label="目标目录")
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=f".{p.name}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fd = -1  # 已移交
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
        tmp_name = ""
        return p
    except OSError as exc:
        raise PathNotWritable(
            f"写入失败：{p}（{type(exc).__name__}: {exc}）",
            hint="确认磁盘未满、目标文件未被其他进程独占（Windows 上被占用时 os.replace 会失败）",
        ) from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    """原子写文本（默认 UTF-8，换行按原样保留 ``\\n``）。"""
    data = text.encode(encoding)
    return atomic_write_bytes(path, data)


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def read_json(path: PathLike, *, default: Any = None, has_default: bool = False) -> Any:
    """编码安全地读 JSON。

    - 文件不存在：``has_default=True`` 时返回 ``default``，否则抛 :class:`PathNotReadable`；
    - 内容非法：抛 :class:`PlatformError`（``code=json_decode_error``），不返回空 dict。
    """
    p = normalize_path(path)
    if not p.exists():
        if has_default:
            return default
        raise PathNotReadable(
            f"JSON 文件不存在：{p}",
            hint="先执行产生该产物的步骤（例如 project_init / dataset_profile）",
        )
    text = read_text_auto(p)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlatformError(
            f"JSON 解析失败：{p}（第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}）",
            code="json_decode_error",
            hint="文件可能被手工编辑破坏或被截断；删除后重跑产生它的步骤",
        ) from exc


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
) -> Path:
    """原子写 JSON（UTF-8 无 BOM，中文不转义，末尾带换行便于 diff）。"""
    try:
        text = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
    except (TypeError, ValueError) as exc:
        raise PlatformError(
            f"JSON 序列化失败：{type(exc).__name__}: {exc}",
            code="json_encode_error",
            hint="对象里含不可序列化类型（numpy 标量/Path/自定义对象），先转成 str/float/list",
        ) from exc
    return atomic_write_text(path, text + "\n")


# --------------------------------------------------------------------------- #
# 文件锁
# --------------------------------------------------------------------------- #

#: 进程内可重入计数：``{线程标识: {锁路径(已解析): 持有深度}}``。
#:
#: ★ 为什么要计数器
#: ``filelock`` 的锁是**操作系统级**的，同一进程里对同一路径再取一次（哪怕是另一个
#: ``FileLock`` 实例）会阻塞到超时 —— 而平台层的组合调用天然可能出现嵌套：
#: ``update_manifest`` 已持锁，mutator 里再调 ``read/write`` 相关函数。因此这里按
#: 「线程 + 解析后的锁路径」计数：同一线程的嵌套直接复用外层锁，跨线程/跨进程仍走
#: 真实文件锁（互斥语义不变）。
_LOCK_DEPTHS: "threading.local" = threading.local()


def _thread_lock_depths() -> Dict[str, int]:
    depths = getattr(_LOCK_DEPTHS, "depths", None)
    if depths is None:
        depths = {}
        _LOCK_DEPTHS.depths = depths
    return depths


@contextmanager
def file_lock(
    path: PathLike,
    *,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
) -> Iterator[Path]:
    """跨平台互斥锁（``filelock``）；超时抛 :class:`LockTimeout`。

    ★ 为什么锁文件与目标文件分离
      锁文件（``*.lock``）只是协调用的 0 字节文件，``os.replace`` 换掉目标
      文件不会影响已持有的锁，避免「换文件把锁换丢」。

    ★ 不用「拿不到锁就继续」
      并发写 manifest 会互相覆盖（lost update）。早期版本为此专门在启动处
      加实例锁；这里保留同样的立场：拿不到锁 = 明确失败。

    ★ 同线程可重入
      同一线程对**同一路径**的嵌套 ``with file_lock(...)`` 复用外层锁（计数式），
      不会自锁超时；内层退出只减计数，外层仍持有。跨线程/跨进程仍互斥。
    """
    if not _HAS_FILELOCK:
        raise PlatformError(
            "缺少跨平台锁依赖 filelock",
            code="dependency_missing",
            hint="执行：python -m pip install 'filelock>=3.12'",
        )
    lock_path = normalize_path(path)
    key = lock_path.as_posix()
    depths = _thread_lock_depths()
    if depths.get(key, 0) > 0:
        # 同线程嵌套：外层已持有，直接复用，不重复向 OS 要锁
        depths[key] += 1
        try:
            yield lock_path
        finally:
            depths[key] -= 1
            if depths[key] <= 0:
                depths.pop(key, None)
        return
    ensure_writable_dir(lock_path.parent, label="锁文件目录")
    lock = FileLock(str(lock_path), timeout=timeout)  # type: ignore[operator]
    try:
        lock.acquire(timeout=timeout, poll_interval=poll_interval)
    except _FileLockTimeout as exc:  # type: ignore[misc]
        raise LockTimeout(
            f"等待文件锁超时（{timeout}s）：{lock_path}",
            hint="另一个建模进程正在写同一项目目录；等它结束，或确认无残留进程后删除该 .lock 文件",
        ) from exc
    depths[key] = 1
    try:
        yield lock_path
    finally:
        depths.pop(key, None)
        try:
            lock.release()
        except Exception:  # noqa: BLE001 - 释放失败不应掩盖业务异常
            pass
