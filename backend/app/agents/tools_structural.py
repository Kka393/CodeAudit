"""
agents/tools_structural.py
=============================
Tools available to the Structural Agent (Agent B).

This agent must reason ONLY from structural/complexity evidence --
never raw source text or comments. Every tool here returns numeric
or structural facts, never free-text code/comment content.
"""

import ast
import re
from typing import List

from app.ml.code_splitter import CodeViews


class StructuralToolset:
    def __init__(self, views: CodeViews, language: str = None):
        self.views = views
        self.language = (language or "").lower()
        self._log: List[dict] = []

    def _record(self, name: str, output):
        self._log.append({"tool": name, "output": output})
        return output

    def get_ast_stats(self) -> dict:
        """Returns max_nesting_depth, cyclomatic_complexity,
        num_functions, avg_function_length. Uses Python AST where
        parsing succeeds; falls back to a line-count/brace-based
        heuristic for other languages or on parse failure. Never
        crashes."""
        if self.language in ("", "python", "py"):
            try:
                stats = self._ast_stats_python()
                return self._record("get_ast_stats", stats)
            except (SyntaxError, ValueError):
                pass

        stats = self._ast_stats_fallback()
        return self._record("get_ast_stats", stats)

    def _ast_stats_python(self) -> dict:
        tree = ast.parse(self.views.main_code)

        def nesting_depth(node, depth=0):
            child_depths = [depth]
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                                       ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_depths.append(nesting_depth(child, depth + 1))
                else:
                    child_depths.append(nesting_depth(child, depth))
            return max(child_depths)

        max_nesting_depth = nesting_depth(tree)

        branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp)
        cyclomatic_complexity = 1 + sum(1 for n in ast.walk(tree) if isinstance(n, branch_nodes))

        func_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        num_functions = len(func_nodes)

        if func_nodes:
            lengths = []
            for f in func_nodes:
                segment = ast.get_source_segment(self.views.main_code, f) or ""
                lengths.append(len(segment.splitlines()))
            avg_function_length = sum(lengths) / len(lengths)
        else:
            avg_function_length = 0.0

        return {
            "max_nesting_depth": max_nesting_depth,
            "cyclomatic_complexity": cyclomatic_complexity,
            "num_functions": num_functions,
            "avg_function_length": round(avg_function_length, 2),
        }

    def _ast_stats_fallback(self) -> dict:
        lines = self.views.main_code.splitlines()
        max_indent = 0
        for line in lines:
            stripped = line.lstrip(" \t")
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent // 4 if indent else 0)

        branch_keywords = re.findall(r"\b(if|for|while|catch|switch|&&|\|\|)\b", self.views.main_code)
        cyclomatic_complexity = 1 + len(branch_keywords)

        num_functions = len(self.views.functions_list)
        if num_functions:
            avg_function_length = sum(len(f.splitlines()) for f in self.views.functions_list) / num_functions
        else:
            avg_function_length = 0.0

        return {
            "max_nesting_depth": max_indent,
            "cyclomatic_complexity": cyclomatic_complexity,
            "num_functions": num_functions,
            "avg_function_length": round(avg_function_length, 2),
        }

    def get_function_signatures(self) -> List[str]:
        """Returns function/method headers only -- no bodies."""
        signatures: List[str] = []

        if self.language in ("", "python", "py"):
            try:
                tree = ast.parse(self.views.main_code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = ", ".join(a.arg for a in node.args.args)
                        signatures.append(f"def {node.name}({args})")
                return self._record("get_function_signatures", signatures)
            except (SyntaxError, ValueError):
                pass

        for func in self.views.functions_list:
            first_line = func.splitlines()[0] if func.splitlines() else func
            signatures.append(first_line.strip())

        return self._record("get_function_signatures", signatures)

    def get_control_flow_summary(self) -> dict:
        """Returns counts of if/for/while/try across extracted
        functions (falls back to whole main_code if no functions were
        extracted)."""
        source_blocks = self.views.functions_list or [self.views.main_code]
        combined = "\n".join(source_blocks)

        counts = {
            "if": len(re.findall(r"\bif\b", combined)),
            "for": len(re.findall(r"\bfor\b", combined)),
            "while": len(re.findall(r"\bwhile\b", combined)),
            "try": len(re.findall(r"\btry\b", combined)),
        }
        return self._record("get_control_flow_summary", counts)

    def compare_complexity_to_typical(self, metric_name: str, value: float) -> float:
        """
        Deterministic 0-1 heuristic scoring how far `value` sits from
        a fixed reference range for `metric_name`.

        NOTE: reference ranges below are explicit PLACEHOLDERS, not
        empirically validated thresholds for this task.
        """
        reference_ranges = {
            "max_nesting_depth": (1, 3),
            "cyclomatic_complexity": (1, 8),
            "num_functions": (1, 5),
            "avg_function_length": (3, 20),
        }

        low, high = reference_ranges.get(metric_name, (0, 10))
        if low <= value <= high:
            score = 1.0
        else:
            distance = min(abs(value - low), abs(value - high))
            score = max(0.0, 1.0 - distance / max(high, 1))

        return self._record(
            "compare_complexity_to_typical", round(min(max(score, 0.0), 1.0), 4)
        )

    @property
    def call_log(self) -> List[dict]:
        return self._log
