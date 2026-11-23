"""受限字段表达式解析器 —— **不使用 eval**，用于样本划分的字段条件。

支持语法
--------
=====================================  ====================================
写法                                    语义
=====================================  ====================================
``month == '2026-01'``                  相等（``!=`` 为不等）
``amount >= 1000``                      ``>`` / ``>=`` / ``<`` / ``<=``
``month in ('2026-01','2026-02')``      集合包含（必须是非空字面量列表）
``channel not in ('app')``              集合不包含（等价 ``~(...)``）
``a >= 1 & b == 'x'``                   与（比 ``|`` 优先级高）
``a == 1 | b == 2``                     或
``(a == 1 | b == 2) & c != ''``         括号
``true`` / ``false`` / ``null``         布尔与空值字面量
``\\`列名带点\\` == 1``                  反引号包住非常规列名
=====================================  ====================================

比较两侧都可以是「列」或「字面量」（``1000 <= amount`` 合法）。

★ 为什么必须自研解析器
----------------------
样本划分的表达式来自用户输入。用 ``eval`` / ``df.query``（内部同样走
``numexpr``/``eval``）会让「表达式」变成任意代码执行入口 ——
``__import__('os').system(...)`` 这类输入必须被**结构性**拒绝，
而不是靠黑名单过滤（黑名单永远漏）。

⇒ 本模块的做法：**先词法分析，只认白名单 token；再递归下降解析成 AST；
最后用 pandas 算子求值**。未识别的字符 / 语法 / 名字（除列名外）一律
抛 :class:`~risklab_mdl.core.errors.ConfigError`，且消息里带位置与示例。

★ 明确拒绝的写法（测试覆盖）
--------------------------
* ``__import__('os')``、``open(...)`` 等函数调用（无调用语法）；
* ``a.b`` 属性访问（``.`` 只在反引号列名内合法）；
* ``a = 1`` 赋值（单个 ``=`` 不是算子，报错并提示用 ``==``）；
* ``;``、``[``、``]``、``lambda``、``__`` 开头标识符、超长表达式（> 2000 字符）。

★ 缺失值的语义
-------------
与 pandas 一致：``NaN != 1`` 为 ``True``（``!=`` 对 NaN 返回真），
``NaN > 1`` 为 ``False``。要显式剔缺失请写 ``x != null``
（``!= null`` 等价 ``x.notna()``，``== null`` 等价 ``x.isna()``）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import ConfigError

__all__ = [
    "MAX_EXPRESSION_LENGTH",
    "Expression",
    "compile_expression",
    "evaluate_expression",
    "expression_columns",
]

#: 表达式长度上限：防止超长输入拖垮词法分析（不是安全边界，只是防呆）
MAX_EXPRESSION_LENGTH = 2000

_COMPARISON_OPS = ("==", "!=", ">=", "<=", ">", "<")

_EXAMPLES = (
    "示例：month in ('2026-01','2026-02') / amount >= 1000 / "
    "(channel == 'app' | channel == 'web') & score != null"
)

_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<op><=|>=|==|!=|>|<)
  | (?P<and>&)
  | (?P<or>\|)
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<comma>,)
  | (?P<number>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
  | (?P<backtick>`[^`]+`)
  | (?P<identifier>[A-Za-z_\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    pos: int


def _fail(text: str, pos: int, message: str) -> "ConfigError":
    return ConfigError(
        f"字段表达式非法（位置 {pos}）：{message}\n"
        f"★ 原表达式：{text!r}\n"
        f"★ 只支持 == != > >= < <= in / not in / & / | / 括号 / 数值与字符串字面量。\n"
        f"★ {_EXAMPLES}"
    )


def _tokenize(text: str) -> list[_Token]:
    if not isinstance(text, str):
        raise ConfigError(f"字段表达式必须是字符串，收到 {type(text).__name__}")
    body = text.strip()
    if not body:
        raise ConfigError(f"字段表达式为空。★ {_EXAMPLES}")
    if len(body) > MAX_EXPRESSION_LENGTH:
        raise ConfigError(
            f"字段表达式过长（{len(body)} > {MAX_EXPRESSION_LENGTH} 字符），请拆分条件。"
        )

    tokens: list[_Token] = []
    pos = 0
    while pos < len(body):
        m = _TOKEN_RE.match(body, pos)
        if m is None:
            ch = body[pos]
            if ch == "=":
                raise _fail(body, pos, "单个 '=' 不是比较符，请改写为 '==' ")
            if ch in ".;[]{}~!@#$%^*\\/":
                raise _fail(
                    body,
                    pos,
                    f"出现不允许的字符 {ch!r}（不支持函数调用 / 属性访问 / 赋值 / 通配符）",
                )
            raise _fail(body, pos, f"无法识别的字符 {ch!r}")
        kind = m.lastgroup or ""
        tok = _Token(kind, m.group(), pos)
        pos = m.end()
        if kind == "space":
            continue
        tokens.append(tok)
    if not tokens:
        raise ConfigError(f"字段表达式为空。★ {_EXAMPLES}")
    return tokens


class _Parser:
    """递归下降：or → and → atom → comparison。"""

    def __init__(self, text: str, tokens: Sequence[_Token]) -> None:
        self.text = text
        self.tokens = list(tokens)
        self.i = 0

    # -- token 工具 -------------------------------------------------------
    def _peek(self) -> _Token | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise _fail(self.text, len(self.text), "表达式意外结束")
        self.i += 1
        return tok

    def _at_end(self) -> bool:
        return self.i >= len(self.tokens)

    # -- 语法 -------------------------------------------------------------
    def parse(self) -> tuple:
        node = self._parse_or()
        if not self._at_end():
            tok = self._peek()
            assert tok is not None
            raise _fail(self.text, tok.pos, f"多余的记号 {tok.text!r}")
        return node

    def _parse_or(self) -> tuple:
        node = self._parse_and()
        while True:
            tok = self._peek()
            if tok is None or tok.kind != "or":
                return node
            self._next()
            right = self._parse_and()
            node = ("or", node, right)

    def _parse_and(self) -> tuple:
        node = self._parse_atom()
        while True:
            tok = self._peek()
            if tok is None or tok.kind != "and":
                return node
            self._next()
            right = self._parse_atom()
            node = ("and", node, right)

    def _parse_atom(self) -> tuple:
        tok = self._peek()
        if tok is None:
            raise _fail(self.text, len(self.text), "表达式不完整")
        if tok.kind == "lparen":
            self._next()
            node = self._parse_or()
            close = self._peek()
            if close is None or close.kind != "rparen":
                raise _fail(self.text, tok.pos, "括号没有闭合")
            self._next()
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> tuple:
        left = self._parse_operand()
        tok = self._peek()
        if tok is None:
            raise _fail(self.text, len(self.text), "缺少比较符（如 == / >= / in）")
        if tok.kind == "identifier" and tok.text.lower() in ("in", "not"):
            negate = False
            if tok.text.lower() == "not":
                self._next()
                nxt = self._peek()
                if nxt is None or nxt.kind != "identifier" or nxt.text.lower() != "in":
                    raise _fail(self.text, tok.pos, "'not' 只能与 'in' 搭配为 'not in'")
                negate = True
            self._next()  # 吃掉 in
            values = self._parse_value_list()
            return ("notin" if negate else "in", left, values)
        if tok.kind != "op":
            raise _fail(
                self.text,
                tok.pos,
                f"期望比较符（== != > >= < <=）或 in，实际读到 {tok.text!r}",
            )
        self._next()
        right = self._parse_operand()
        return ("cmp", tok.text, left, right)

    def _parse_value_list(self) -> tuple:
        tok = self._peek()
        if tok is None or tok.kind != "lparen":
            raise _fail(
                self.text,
                tok.pos if tok else len(self.text),
                "'in' 后面必须跟括号包住的字面量列表，如 in ('a','b')",
            )
        self._next()
        values: list[tuple] = []
        while True:
            tok = self._peek()
            if tok is None:
                raise _fail(self.text, len(self.text), "'in (...)' 缺少右括号")
            if tok.kind == "rparen":
                self._next()
                break
            values.append(self._parse_literal())
            tok = self._peek()
            if tok is not None and tok.kind == "comma":
                self._next()
                continue
            tok = self._peek()
            if tok is None or tok.kind != "rparen":
                raise _fail(
                    self.text,
                    tok.pos if tok else len(self.text),
                    "'in (...)' 中的字面量之间需要逗号或右括号",
                )
        if not values:
            raise _fail(self.text, tok.pos, "'in ()' 列表不能为空")
        return tuple(values)

    def _parse_operand(self) -> tuple:
        tok = self._peek()
        if tok is None:
            raise _fail(self.text, len(self.text), "表达式不完整")
        if tok.kind in ("number", "string", "backtick", "identifier"):
            if tok.kind == "identifier" and tok.text.lower() in (
                "in",
                "not",
                "and",
                "or",
            ):
                raise _fail(self.text, tok.pos, f"保留字 {tok.text!r} 不能作为列名")
            self._next()
            if tok.kind == "number":
                return ("lit", _to_number(tok.text))
            if tok.kind == "string":
                return ("lit", _unquote(tok.text))
            if tok.kind == "backtick":
                return ("col", tok.text[1:-1], tok.pos)
            lowered = tok.text.lower()
            if lowered in ("true", "false"):
                return ("lit", lowered == "true")
            if lowered in ("null", "none", "nan"):
                return ("lit", None)
            if tok.text.startswith("__"):
                raise _fail(self.text, tok.pos, "列名不允许以 '__' 开头")
            return ("col", tok.text, tok.pos)
        raise _fail(
            self.text, tok.pos, f"此处需要列名或字面量，实际读到 {tok.text!r}"
        )

    def _parse_literal(self) -> tuple:
        node = self._parse_operand()
        if node[0] != "lit":
            raise _fail(
                self.text,
                self._peek().pos if self._peek() else len(self.text),
                "'in (...)' 列表里只允许字面量，不允许列名",
            )
        return node


def _to_number(text: str) -> float | int:
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError as exc:  # pragma: no cover - 正则已保证可转
        raise ConfigError(f"数字字面量非法：{text!r}") from exc


def _unquote(text: str) -> str:
    quote = text[0]
    body = text[1:-1]
    body = body.replace("\\" + quote, quote).replace("\\\\", "\\")
    return body


# --------------------------------------------------------------------------- #
# 求值
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Expression:
    """编译后的表达式（不可变，可复用于不同数据框）。"""

    text: str
    ast: tuple
    columns: tuple[str, ...]

    def __str__(self) -> str:
        return self.text

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """在 ``df`` 上求值，返回与 ``df`` 同索引的 bool Series。"""
        if not isinstance(df, pd.DataFrame):
            raise ConfigError(f"表达式求值需要 DataFrame，收到 {type(df).__name__}")
        mask = _eval(self.ast, df, self.text)
        if isinstance(mask, pd.Series):
            return mask.reindex(df.index).fillna(False).astype(bool)
        # 常量表达式（如 ``true``）⇒ 全行同一取值
        return pd.Series(bool(mask), index=df.index, dtype="bool")


def _eval(node: tuple, df: pd.DataFrame, text: str):
    kind = node[0]
    if kind == "or":
        return _eval(node[1], df, text) | _eval(node[2], df, text)
    if kind == "and":
        return _eval(node[1], df, text) & _eval(node[2], df, text)
    if kind == "in" or kind == "notin":
        left = _value(node[1], df, text)
        values = [n[1] for n in node[2]]
        if isinstance(left, pd.Series):
            mask = left.isin(values)
        else:
            mask = pd.Series(left in values, index=df.index)
        return mask if kind == "in" else ~mask
    if kind == "cmp":
        op = node[1]
        left = _value(node[2], df, text)
        right = _value(node[3], df, text)
        return _compare(op, left, right, text)
    raise ConfigError(f"内部错误：未知表达式节点 {kind!r}")  # pragma: no cover


def _value(node: tuple, df: pd.DataFrame, text: str):
    if node[0] == "lit":
        return node[1]
    name = node[1]
    if name not in df.columns:
        raise ConfigError(
            f"字段表达式引用了不存在的列 {name!r}。\n"
            f"★ 原表达式：{text!r}\n"
            f"★ 数据实际列名（前 20 个）：{list(df.columns)[:20]}"
        )
    return df[name]


def _compare(op: str, left, right, text: str):
    left_is_series = isinstance(left, pd.Series)
    right_is_series = isinstance(right, pd.Series)

    # null 特判：== null / != null 走 isna / notna（NaN 的默认比较语义反直觉）
    if op in ("==", "!="):
        if left is None or right is None:
            series = left if left_is_series else right
            if not isinstance(series, pd.Series):
                return pd.Series(bool((left is None) == (right is None)))
            mask = series.isna() if op == "==" else series.notna()
            return mask

    if op in ("==", "!="):
        if left_is_series and isinstance(right, (bool, int, float, str)):
            result = left == right if op == "==" else left != right
        elif right_is_series and isinstance(left, (bool, int, float, str)):
            result = right == left if op == "==" else right != left
        else:
            result = left == right if op == "==" else left != right
        return result if isinstance(result, pd.Series) else pd.Series(result)

    # 数值比较：先把 Series 侧转成数值，转不动就显式报错（不静默全 False）
    lhs = _numeric_side(left, text) if left_is_series else left
    rhs = _numeric_side(right, text) if right_is_series else right
    if not isinstance(lhs, (pd.Series, int, float, np.number)) and lhs is not None:
        raise ConfigError(f"'in' / 比较运算的右侧类型不支持：{type(lhs).__name__}")
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    raise ConfigError(f"不支持的比较符 {op!r}")  # pragma: no cover


def _numeric_side(value, text: str):
    if not isinstance(value, pd.Series):
        return value
    if pd.api.types.is_numeric_dtype(value) and not pd.api.types.is_bool_dtype(value):
        return value.astype(float)
    conv = pd.to_numeric(value, errors="coerce")
    bad = int((conv.isna() & value.notna()).sum())
    if bad:
        raise ConfigError(
            f"列 {getattr(value, 'name', '')!r} 含 {bad} 个非数值取值，"
            f"无法参与 > >= < <= 比较。\n"
            f"★ 原表达式：{text!r}\n"
            f"★ 请改用 == / in 比较，或先把该列清洗为数值。"
        )
    return conv.astype(float)


def _collect_columns(node: tuple, out: list[str], by_name: Mapping[str, Any]) -> None:
    kind = node[0]
    if kind in ("or", "and"):
        _collect_columns(node[1], out, by_name)
        _collect_columns(node[2], out, by_name)
    elif kind in ("in", "notin"):
        _collect_columns(node[1], out, by_name)
    elif kind == "cmp":
        _collect_columns(node[2], out, by_name)
        _collect_columns(node[3], out, by_name)
    elif kind == "col":
        if node[1] not in out:
            out.append(node[1])


def compile_expression(text: str) -> Expression:
    """把用户输入的表达式编译成可复用的 :class:`Expression`。

    Raises
    ------
    ConfigError
        空表达式、非法语法、危险写法（``__import__`` / 属性访问 / 赋值）等。
    """
    tokens = _tokenize(text)
    ast = _Parser(text if isinstance(text, str) else str(text), tokens).parse()
    cols: list[str] = []
    _collect_columns(ast, cols, {})
    return Expression(text=text.strip(), ast=ast, columns=tuple(cols))


def evaluate_expression(text: str, df: pd.DataFrame) -> pd.Series:
    """一次性编译 + 求值（便捷入口）。"""
    return compile_expression(text).evaluate(df)


def expression_columns(text: str) -> tuple[str, ...]:
    """返回表达式引用的列名（供调用方提前校验列是否存在）。"""
    return compile_expression(text).columns
