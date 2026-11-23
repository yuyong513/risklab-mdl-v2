"""stdio 冒烟：真子进程起 MCP 服务，按 JSON-RPC 顺序握手并调用探针工具。

★ 为什么要真子进程 + 顺序交互
----------------------------
``stdio`` 是 MCP 客户端的标准接入方式。一次性灌完所有消息再关 stdin 会撞上
「服务端读到 EOF 提前收摊」的竞态（实测会丢掉最后一个 tools/call 响应），
所以这里按真实客户端的方式逐条收发：发请求 → 等对应 id 的响应 → 再发下一条。

验收 ④ 由本文件承担：``tools/list`` 成功且工具集含 ``server_probe``。
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from risklab_mdl import sdk as sdk_mod

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "risklab-mdl-v2-tests", "version": "2.0.0"}
SESSION_TIMEOUT = 180.0


class StdioClient:
    """极简 MCP stdio 客户端（只为冒烟测试服务，不追求完整协议覆盖）。"""

    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "risklab_mdl", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._inbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._pending: Dict[Any, Dict[str, Any]] = {}
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.stderr_text = ""

    # ---------- 内部 ----------

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue  # 非协议帧（例如调试输出）不算响应
            if isinstance(message, dict):
                self._inbox.put(message)

    def _send(self, message: Dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    # ---------- 协议动作 ----------

    def request(self, request_id: int, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        while True:
            try:
                message = self._inbox.get(timeout=SESSION_TIMEOUT)
            except queue.Empty as exc:  # pragma: no cover - 只在服务端卡死时触发
                raise AssertionError(
                    f"等 {method}（id={request_id}）响应超时；stderr=\n{self._drain_stderr()}"
                ) from exc
            if message.get("id") == request_id:
                return message
            self._pending[message.get("id")] = message

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _drain_stderr(self) -> str:
        if self.process.poll() is None or self.stderr_text:
            return self.stderr_text
        assert self.process.stderr is not None
        self.stderr_text = self.process.stderr.read()
        return self.stderr_text

    def close(self) -> str:
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:  # noqa: BLE001 - 清理失败不掩盖断言结果
            self.process.kill()
            self.process.wait(timeout=30)
        assert self.process.stderr is not None
        self.stderr_text = self.process.stderr.read()
        return self.stderr_text


class StdioSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = StdioClient()
        try:
            init = self.client.request(
                1,
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO},
            )
            self.init_result = init.get("result") or {}
            self.client.notify("notifications/initialized")
        except Exception:
            self.client.close()
            raise

    def tearDown(self) -> None:
        self.client.close()

    def test_initialize_returns_server_info(self) -> None:
        self.assertIn("serverInfo", self.init_result, f"initialize 缺 serverInfo：{self.init_result}")
        server_info = self.init_result["serverInfo"]
        self.assertIn("risklab", server_info.get("name", ""))
        self.assertEqual(server_info.get("version"), "2.0.0")
        self.assertIn("tools", self.init_result.get("capabilities", {}))

    def test_tools_list_contains_server_probe(self) -> None:
        response = self.client.request(2, "tools/list")
        result = response.get("result") or {}
        tools: List[Dict[str, Any]] = result.get("tools") or []
        names = [tool["name"] for tool in tools]
        self.assertIn("server_probe", names, f"tools/list 未包含 server_probe，实际：{names}")
        # T3：数据/特征分析 6 个；T4：预处理/划分/筛选 5 个；T5：模型/过程 8 个
        # ⇒ 合计 20 个工具（server_probe + 6 + 5 + 8）
        expected = [
            "server_probe",
            "project_init",
            "dataset_load",
            "dataset_columns",
            "dataset_profile",
            "feature_analyze",
            "feature_charts",
            "preprocess_check",
            "preprocess_apply",
            "dataset_split",
            "feature_prescreen",
            "feature_refine",
            "model_catalog",
            "model_train",
            "model_tune",
            "model_evaluate",
            "run_list",
            "run_compare",
            "model_confirm",
            "report_build",
        ]
        self.assertEqual(len(expected), 20, "T5 末工具集应为 server_probe + 6 + 5 + 8 = 20")
        self.assertEqual(sorted(names), sorted(expected), f"工具集合不符预期，实际：{names}")
        by_name = {tool["name"]: tool for tool in tools}
        probe = by_name["server_probe"]
        self.assertTrue(probe.get("description"), "工具必须有描述（供 agent 选择）")
        for tool in tools:
            self.assertTrue(tool.get("description"), f"{tool['name']} 缺少描述")
            self.assertEqual(tool.get("inputSchema", {}).get("type", "object"), "object")

    def test_server_probe_call_returns_environment_report(self) -> None:
        response = self.client.request(3, "tools/call", {"name": "server_probe", "arguments": {}})
        result = response.get("result") or {}
        self.assertFalse(result.get("isError", False), f"探针调用报错：{result}")
        payload = extract_payload(result)
        for key in ("ok", "status", "data", "version", "platform", "deps", "artifacts_check"):
            self.assertIn(key, payload, f"探针返回缺少 {key}：{payload}")
        self.assertTrue(payload["ok"], f"本机探针应通过：{payload.get('error')}")
        self.assertEqual(payload["version"], "2.0.0")
        self.assertTrue(payload["artifacts_check"]["ok"], payload["artifacts_check"].get("error"))
        self.assertIn("python", payload["platform"])
        self.assertIn("mcp", payload["deps"])
        self.assertIsNone(payload["deps"]["mcp"]["error"], "本机 mcp 应可用")

    def test_server_probe_data_mirrors_flat_fields(self) -> None:
        """通用协议 §1 的 data 必须存在，且与 T1 已发布的扁平字段内容一致。"""
        response = self.client.request(4, "tools/call", {"name": "server_probe", "arguments": {}})
        payload = extract_payload(response.get("result") or {})
        data = payload.get("data")
        self.assertIsInstance(data, dict, f"data 缺失或类型错误：{payload}")
        for key in ("version", "platform", "deps", "artifacts_check"):
            self.assertIn(key, data, f"data 缺少 {key}")
            self.assertEqual(data[key], payload[key], f"data.{key} 与顶层 {key} 不一致")
        self.assertEqual(payload.get("status"), "ok")


class SdkAdapterTests(unittest.TestCase):
    """I5：``make_server`` 必须同时兼容 2.x（接受 version）与 1.x（拒绝 version）。"""

    def _resolve_to(self, cls: Any) -> Any:
        resolved = sdk_mod.ServerClass(module="fake.module", name="Fake", cls=cls)
        return mock.patch.object(sdk_mod, "resolve_server_class", return_value=resolved)

    def test_make_server_passes_version_when_class_accepts_it(self) -> None:
        captured: Dict[str, Any] = {}

        class AcceptsVersion:
            def __init__(self, name: str, version: str = "") -> None:
                captured["name"] = name
                captured["version"] = version
                self.name = name

        with self._resolve_to(AcceptsVersion):
            server = sdk_mod.make_server("risklab-mdl-v2", "2.0.0")
        self.assertIsInstance(server, AcceptsVersion)
        self.assertEqual(captured, {"name": "risklab-mdl-v2", "version": "2.0.0"})

    def test_make_server_falls_back_for_1x_class_rejecting_version(self) -> None:
        calls: List[Dict[str, Any]] = []

        class OldFastMCP:
            """模拟 1.x FastMCP：只接受 name，多给 version 就 TypeError。"""

            def __init__(self, name: str, **kwargs: Any) -> None:
                calls.append(dict(kwargs))
                if "version" in kwargs:
                    raise TypeError("__init__() got an unexpected keyword argument 'version'")
                self.name = name

        with self._resolve_to(OldFastMCP):
            server = sdk_mod.make_server("risklab-mdl-v2", "2.0.0")
        self.assertEqual(len(calls), 2, "应先尝试带 version，再回退最小签名")
        self.assertEqual(calls[0], {"version": "2.0.0"}, "第一次必须按 2.x 签名构造")
        self.assertEqual(calls[1], {}, "回退时不得再传 version")
        self.assertEqual(server.name, "risklab-mdl-v2")


def extract_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """工具的 dict 返回值可能落在 structuredContent 或 text 内容里。"""
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and structured:
        # 部分 SDK 会包一层 {"result": {...}}
        if set(structured) == {"result"} and isinstance(structured["result"], dict):
            return structured["result"]
        return structured
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            try:
                parsed = json.loads(item.get("text", ""))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise AssertionError(f"无法从工具结果中取出 JSON 载荷：{result}")


if __name__ == "__main__":
    unittest.main()
