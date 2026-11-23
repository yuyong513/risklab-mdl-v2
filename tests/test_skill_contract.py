"""T6 验收 ①③：skill 包契约 —— 20 工具覆盖、旧交互关键词清零、三份 MCP 配置可握手。

三段断言对应三条硬要求：

① **工具覆盖 100%**：工具名不是手抄的，而是从 `server.build_server().list_tools()`
   现取（20 个）；`skill/references/` 里每个工具**有且仅有**一段
   「何时调用 / 入参示例 / 结果解读」，用 `<!-- tool: <名字> -->` 标记定位，
   段内必须同时出现三段小标题和至少一个代码块。
   多一个标记（比如把某个工具写两遍）或少一个工具，测试都会红。

② **旧交互关键词清零**：`skill/` 目录下**任何文件**都不许出现旧项目的交互残留
   （`弹窗` / `参数文档` / `flow_run_step` / `flow_get_prompt` / `HTML 卡片`）——
   这些是「服务端渲染界面」时代的做法，本项目只返回数据载荷。

③ **MCP 配置可用**：`mcp/mcp.*.json` 三份都是合法 JSON、含 `mcpServers`，
   并且用配置里的 `command`/`args` 真起一个 stdio 子进程做
   `initialize` + `tools/list` 握手，返回的工具集合与 `list_tools()` 完全一致。
   ★ 这一条证明的是「配置内容可用」，不是「某个客户端界面能加载它」——
   客户端侧实测范围见 `docs/skill使用指南.md` §2。
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from risklab_mdl.server import build_server

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skill"
REFERENCES_DIR = SKILL_DIR / "references"
MCP_DIR = ROOT / "mcp"
SKILL_MD = SKILL_DIR / "SKILL.md"

#: 段标记：`<!-- tool: <名字> -->`
MARKER_PATTERN = re.compile(r"<!--\s*tool:\s*([A-Za-z_][A-Za-z0-9_]*)\s*-->")
#: 每段必须出现的三段小标题
REQUIRED_SECTIONS = ("何时调用", "入参示例", "结果解读")
#: 旧项目交互残留关键词（服务端不再渲染任何界面）
LEGACY_KEYWORDS = ("弹窗", "参数文档", "flow_run_step", "flow_get_prompt", "HTML 卡片")

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "risklab-mdl-v2-t6-tests", "version": "2.0.0"}
SESSION_TIMEOUT = 180.0


def registered_tool_names() -> List[str]:
    """从真实服务器实例取工具名（唯一真源，不手抄）。"""
    server = build_server()
    tools = asyncio.run(server.list_tools())
    return [tool.name for tool in tools]


def iter_marker_blocks() -> List[tuple]:
    """把 `references/*.md` 按标记切成 ``[(文件名, 工具名, 段正文), ...]``。"""
    blocks: List[tuple] = []
    for path in sorted(REFERENCES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        parts = MARKER_PATTERN.split(text)
        # parts = [前导, 名字1, 正文1, 名字2, 正文2, ...]
        for index in range(1, len(parts), 2):
            blocks.append((path.name, parts[index], parts[index + 1]))
    return blocks


# --------------------------------------------------------------------------- #
# ① 20/20 工具覆盖
# --------------------------------------------------------------------------- #
def test_skill_md_frontmatter_and_paths() -> None:
    """SKILL.md 必须存在、frontmatter 有 name/description、references 全在。"""
    assert SKILL_MD.is_file(), f"缺少 {SKILL_MD}"
    text = SKILL_MD.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md 必须以 YAML frontmatter 开头"
    frontmatter = text.split("---\n", 2)[1]
    assert re.search(r"^name:\s*risklab-mdl\s*$", frontmatter, re.M), frontmatter[:200]
    description = re.search(r"^description:\s*(.+)$", frontmatter, re.M)
    assert description, "frontmatter 缺 description（何时触发）"
    # 触发条件要覆盖三类场景
    for keyword in ("CSV", "信用风险", "特征分析", "模型"):
        assert keyword in description.group(1), f"description 未覆盖触发场景：{keyword}"

    for name in ("tools.md", "workflow.md", "models.md", "metrics.md", "errors.md", "artifacts.md"):
        assert (REFERENCES_DIR / name).is_file(), f"缺少 references/{name}"

    # 9 环节 + 交互协议 + 边界三块内容必须在正文里
    for heading in ("9 环节", "needs_user_input", "边界"):
        assert heading in text, f"SKILL.md 缺少：{heading}"
    for stage in (
        "数据上传与项目引导",
        "样本数据分析",
        "特征分析",
        "模型选型",
        "特征筛选",
        "入模前检查与特征精调",
        "样本划分",
        "模型训练",
        "模型结果确认",
    ):
        assert stage in text, f"SKILL.md 缺少环节：{stage}"


def test_all_twenty_tools_covered_exactly_once() -> None:
    """20 个工具在 references 中各有一段「何时调用 / 入参示例 / 结果解读」。"""
    names = registered_tool_names()
    assert len(names) == 20, f"服务端工具数应为 20，实际 {len(names)}：{names}"
    assert len(set(names)) == 20, names

    blocks = iter_marker_blocks()
    counts = Counter(tool for _, tool, _ in blocks)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    assert not duplicates, f"这些工具被写了两段（要求有且仅有一段）：{duplicates}"

    missing = sorted(set(names) - set(counts))
    extra = sorted(set(counts) - set(names))
    assert not missing, f"references 缺这些工具的段落：{missing}"
    assert not extra, f"references 里有未注册工具的段落：{extra}"

    for filename, tool, body in blocks:
        for section in REQUIRED_SECTIONS:
            assert section in body, f"{filename} / {tool}：缺「{section}」"
        assert "```" in body, f"{filename} / {tool}：入参示例必须是代码块"
        assert len(body.strip()) > 200, f"{filename} / {tool}：段落太短，不足以指导 agent"


def test_references_mention_real_tool_arguments() -> None:
    """入参示例里必须出现真实参数名（防止文档与实现漂移）。"""
    text = "\n".join(path.read_text(encoding="utf-8") for path in REFERENCES_DIR.glob("*.md"))
    for arg in ("csv_path", "project_dir", "run_id", "config", "spec", "thresholds", "mode"):
        assert f'"{arg}"' in text, f"references 里没有出现过真实入参名：{arg}"
    # 关键默认值/口径必须写清（用户会照着抄）
    for token in ("0.25", "0.10", "/mdl-", "fingerprint", "-inf"):
        assert token in text, f"references 里缺少关键约定：{token}"


# --------------------------------------------------------------------------- #
# ② 旧交互关键词清零
# --------------------------------------------------------------------------- #
def test_no_legacy_interaction_keywords() -> None:
    """`skill/` 下任何文件都不得出现旧交互残留关键词。"""
    offenders: List[str] = []
    files = [path for path in SKILL_DIR.rglob("*") if path.is_file()]
    assert files, "skill 目录为空"
    for path in files:
        text = path.read_text(encoding="utf-8")
        for keyword in LEGACY_KEYWORDS:
            if keyword in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()} 含 “{keyword}”")
    assert not offenders, "旧交互关键词残留：\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# ③ 三份 MCP 配置样例
# --------------------------------------------------------------------------- #
def load_mcp_configs() -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    for name in ("mcp.claude.json", "mcp.dsh.json", "mcp.workbuddy.json"):
        path = MCP_DIR / name
        assert path.is_file(), f"缺少 {path}"
        raw = path.read_text(encoding="utf-8")
        config = json.loads(raw)  # 非法 JSON 直接抛
        configs[name] = config
    return configs


def test_mcp_configs_are_valid_json_with_stdio_server() -> None:
    configs = load_mcp_configs()
    for name, config in configs.items():
        assert isinstance(config, dict), name
        assert "mcpServers" in config, f"{name}: 缺 mcpServers"
        servers = config["mcpServers"]
        assert isinstance(servers, dict) and servers, f"{name}: mcpServers 不能为空"
        entry = servers.get("risklab-mdl")
        assert isinstance(entry, dict), f"{name}: 缺 risklab-mdl 条目"
        assert isinstance(entry.get("command"), str) and entry["command"], f"{name}: 缺 command"
        args = entry.get("args")
        assert isinstance(args, list) and args and all(isinstance(a, str) for a in args), (
            f"{name}: args 必须是字符串数组"
        )
        # stdio 接入：command + args 必须能定位到本包（`python -m risklab_mdl` 或 console script）
        joined = " ".join([entry["command"], *args])
        assert "risklab_mdl" in joined or "risklab-mdl" in joined, f"{name}: 未指向本包：{joined}"
        assert "transport" not in joined or "stdio" in joined, f"{name}: 默认必须是 stdio"


class _StdioClient:
    """极简 stdio MCP 客户端：只为「配置能不能真跑起来」服务。"""

    def __init__(self, command: str, args: List[str]) -> None:
        self.process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._inbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        reader = threading.Thread(target=self._read_loop, daemon=True)
        reader.start()
        self.stderr_text = ""

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._inbox.put(message)

    def _send(self, message: Dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def request(self, request_id: int, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        while True:
            try:
                message = self._inbox.get(timeout=SESSION_TIMEOUT)
            except queue.Empty as exc:  # pragma: no cover - 只在服务端卡死时触发
                raise AssertionError(f"等 {method} 响应超时；stderr=\n{self._drain_stderr()}") from exc
            if message.get("id") == request_id:
                return message

    def notify(self, method: str, params: Dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _drain_stderr(self) -> str:
        if self.stderr_text:
            return self.stderr_text
        if self.process.poll() is not None and self.process.stderr is not None:
            self.stderr_text = self.process.stderr.read()
        return self.stderr_text

    def close(self) -> str:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:  # noqa: BLE001 - 清理失败不掩盖断言
            self.process.kill()
            self.process.wait(timeout=30)
        assert self.process.stderr is not None
        self.stderr_text = self.process.stderr.read()
        return self.stderr_text


def test_mcp_config_samples_handshake() -> None:
    """用三份配置里的 command/args 真握手：initialize + tools/list 必须给出 20 个工具。"""
    expected = sorted(registered_tool_names())
    configs = load_mcp_configs()
    for name, config in configs.items():
        entry = config["mcpServers"]["risklab-mdl"]
        command, args = entry["command"], list(entry["args"])
        resolved = shutil.which(command)
        if resolved is None:
            # 只有 command 是解释器别名时才允许回退（CI/无 PATH 环境）；其余情况必须失败
            assert Path(command).suffix.lower() in (".exe", "") or command.startswith("python"), (
                f"{name}: PATH 里找不到 command={command!r}"
            )
            resolved = sys.executable
        client = _StdioClient(resolved, args)
        try:
            init = client.request(
                1,
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO},
            )
        except Exception:
            client.close()
            raise
        try:
            result = init.get("result") or {}
            assert "serverInfo" in result, f"{name}: initialize 缺 serverInfo：{init}"
            assert "risklab" in result["serverInfo"].get("name", ""), f"{name}: {result['serverInfo']}"
            client.notify("notifications/initialized")
            tools = (client.request(2, "tools/list").get("result") or {}).get("tools") or []
            names = sorted(tool["name"] for tool in tools)
            assert len(names) == 20, f"{name}: tools/list 应有 20 个工具，实际 {len(names)}：{names}"
            assert names == expected, f"{name}: 工具集合与 list_tools() 不一致：{names}"
            for tool in tools:
                assert tool.get("description"), f"{name}: {tool['name']} 缺 description"
                assert tool.get("inputSchema", {}).get("type", "object") == "object", (
                    f"{name}: {tool['name']} 的 inputSchema 不是 object"
                )
        finally:
            client.close()
