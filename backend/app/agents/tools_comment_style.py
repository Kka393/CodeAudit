"""
agents/tools_comment_style.py
================================
Tools available to the Comment/Style Agent (Agent A).

IMPORTANT: this toolset does NOT re-parse the source code. It wraps
the SAME CodeViews / ParseResult already produced by
ml/code_splitter.py for the ML branch, so the agent's evidence is
guaranteed consistent with what the ML model actually saw.
"""

import ast
import re
from typing import List

from app.ml.code_splitter import CodeViews


class CommentStyleToolset:
    def __init__(self, views: CodeViews, language: str = None):
        self.views = views
        self.language = (language or "").lower()
        self._log: List[dict] = []  # every tool call + raw output, for evidence validation

    def _record(self, name: str, output):
        self._log.append({"tool": name, "output": output})
        return output

    def get_comments(self) -> List[str]:
        """Returns the existing comments_view content split into
        individual comments (one per originally-extracted comment)."""
        return self._record("get_comments", list(self.views.comments_list))

    def get_identifier_names(self) -> dict:
        """Returns function names, variable names, and parameter
        names extracted from main_code and sub_functions. Reuses the
        existing AST extraction for Python; best-effort regex for
        other languages."""
        function_names, variable_names, parameter_names = [], [], []

        if self.language in ("", "python", "py"):
            try:
                tree = ast.parse(self.views.main_code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_names.append(node.name)
                        for arg in node.args.args:
                            parameter_names.append(arg.arg)
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        variable_names.append(node.id)
            except (SyntaxError, ValueError):
                pass
        else:
            # Best-effort regex fallback for non-Python languages.
            function_names = re.findall(r"\b(?:function|func|def)\s+(\w+)", self.views.main_code)
            function_names += re.findall(
                r"\b(?:public|private|protected|static|\s)*[\w<>\[\]]+\s+(\w+)\s*\(", self.views.main_code
            )
            variable_names = re.findall(r"\b(?:var|let|const)\s+(\w+)", self.views.main_code)

        result = {
            "function_names": sorted(set(function_names)),
            "variable_names": sorted(set(variable_names)),
            "parameter_names": sorted(set(parameter_names)),
        }
        return self._record("get_identifier_names", result)

    def get_docstrings(self) -> List[str]:
        """Returns available docstring-like comments. May legally
        return [] -- must never crash when no docstrings exist."""
        docstrings: List[str] = []

        if self.language in ("", "python", "py"):
            try:
                tree = ast.parse(self.views.main_code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef)):
                        doc = ast.get_docstring(node)
                        if doc:
                            docstrings.append(doc.strip())
            except (SyntaxError, ValueError):
                pass

        return self._record("get_docstrings", docstrings)

    def compare_style_to_reference(self, text: str) -> float:
        """
        Deterministic 0-1 heuristic based on sentence-length
        uniformity, punctuation consistency, and capitalization
        consistency.

        NOTE: this is an explicit PLACEHOLDER heuristic, not a
        validated standalone style detector. It exists to give the
        agent a cheap, reproducible signal to reason with -- it is
        not a claim about ground-truth authorship style.
        """
        if not text or not text.strip():
            return 0.5

        sentences = [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]
        if not sentences:
            return 0.5

        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        length_uniformity = 1.0 / (1.0 + variance / 10.0)

        punctuation_ends = sum(1 for s in sentences if s and s[-1] in ".!?")
        punctuation_consistency = punctuation_ends / len(sentences)

        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        capitalization_consistency = capitalized / len(sentences)

        score = (length_uniformity + punctuation_consistency + capitalization_consistency) / 3.0
        return self._record("compare_style_to_reference", round(min(max(score, 0.0), 1.0), 4))

    @property
    def call_log(self) -> List[dict]:
        return self._log
