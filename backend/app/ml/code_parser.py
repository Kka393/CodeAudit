"""
ml/code_parser.py
==================
REUSED, unmodified in behavior from the existing AICD_MultiView
research project's src/preprocessing/code_parser.py: AST-based
extraction for Python, regex-based fallback extraction for every
other supported language. See the original project for full
rationale in comments; kept intact here since it is load-bearing for
both the ML branch (code_splitter -> embeddings) and the Structural
/ Comment-Style agent toolsets, which reuse this SAME parser output
rather than re-parsing independently.
"""

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import List


@dataclass
class ParseResult:
    functions: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    status: str = "unsupported"  # "ast_python" | "regex_fallback" | "unsupported"


_LINE_COMMENT_RE = re.compile(r"(?:^|[^:])//(.*)$|(?:^|\s)#(.*)$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*(.*?)\*/", re.DOTALL)

_FUNC_SIGNATURE_RE = re.compile(
    r"""
    ^[ \t]*
    (?:
        def[ \t]+\w+[ \t]*\([^)]*\)[ \t]*:
        |
        (?:public|private|protected|internal|static|\s)*
        [\w\<\>\[\]]+[ \t]+\w+[ \t]*\([^)]*\)[ \t]*\{?
        |
        func[ \t]+(?:\([^)]*\)[ \t]*)?\w+[ \t]*\([^)]*\)[^\{]*\{
        |
        function[ \t]+\w+[ \t]*\([^)]*\)[ \t]*\{
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


def _parse_python_ast(source_code: str) -> ParseResult:
    tree = ast.parse(source_code)
    functions: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source_code, node)
            if segment:
                functions.append(segment)

    comments: List[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                comments.append(tok.string.lstrip("#").strip())
    except tokenize.TokenizeError:
        pass

    return ParseResult(functions=functions, comments=comments, status="ast_python")


def _extract_brace_matched_block(source_code: str, start_index: int) -> str:
    brace_start = source_code.find("{", start_index)
    if brace_start == -1:
        newline = source_code.find("\n", start_index)
        return source_code[start_index: newline if newline != -1 else len(source_code)]

    depth = 0
    for i in range(brace_start, len(source_code)):
        if source_code[i] == "{":
            depth += 1
        elif source_code[i] == "}":
            depth -= 1
            if depth == 0:
                return source_code[start_index: i + 1]
    return source_code[start_index:]


def _parse_generic_regex(source_code: str) -> ParseResult:
    comments: List[str] = []
    for match in _BLOCK_COMMENT_RE.finditer(source_code):
        text = match.group(1).strip()
        if text:
            comments.append(text)

    code_without_blocks = _BLOCK_COMMENT_RE.sub("", source_code)
    for match in _LINE_COMMENT_RE.finditer(code_without_blocks):
        text = (match.group(1) or match.group(2) or "").strip()
        if text:
            comments.append(text)

    functions: List[str] = []
    for match in _FUNC_SIGNATURE_RE.finditer(source_code):
        block = _extract_brace_matched_block(source_code, match.start())
        if block.strip():
            functions.append(block.strip())

    return ParseResult(functions=functions, comments=comments, status="regex_fallback")


def parse_source(source_code: str, language: str = None) -> ParseResult:
    if not source_code or not source_code.strip():
        return ParseResult(functions=[], comments=[], status="unsupported")

    language_lc = (language or "").lower()

    if language_lc in ("", "python", "py"):
        try:
            return _parse_python_ast(source_code)
        except (SyntaxError, ValueError):
            pass

    return _parse_generic_regex(source_code)
