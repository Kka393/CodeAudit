"""
Tests 5, 6: CommentStyleToolset and StructuralToolset.
"""

from app.agents.tools_comment_style import CommentStyleToolset
from app.agents.tools_structural import StructuralToolset
from app.ml.code_splitter import split_code

SOURCE = '''
def add(a, b):
    """Adds two numbers."""
    # simple addition
    total = a + b
    return total
'''


def test_comment_style_toolset_no_crash_on_no_comments():
    views = split_code("def f(x):\n    return x", language="python")
    toolset = CommentStyleToolset(views, language="python")
    assert toolset.get_comments() == []
    assert toolset.get_docstrings() == []
    score = toolset.compare_style_to_reference("")
    assert 0.0 <= score <= 1.0


def test_comment_style_toolset_extracts_identifiers_and_docstrings():
    views = split_code(SOURCE, language="python")
    toolset = CommentStyleToolset(views, language="python")

    identifiers = toolset.get_identifier_names()
    assert "add" in identifiers["function_names"]
    assert "total" in identifiers["variable_names"]

    docstrings = toolset.get_docstrings()
    assert any("Adds two numbers" in d for d in docstrings)

    comments = toolset.get_comments()
    assert any("simple addition" in c for c in comments)

    # every recorded tool output must be present in the call log for
    # evidence validation to work
    assert len(toolset.call_log) == 3


def test_structural_toolset_ast_stats_and_signatures():
    views = split_code(SOURCE, language="python")
    toolset = StructuralToolset(views, language="python")

    stats = toolset.get_ast_stats()
    assert stats["num_functions"] == 1
    assert stats["cyclomatic_complexity"] >= 1

    signatures = toolset.get_function_signatures()
    assert any("add" in s for s in signatures)

    control_flow = toolset.get_control_flow_summary()
    assert set(control_flow.keys()) == {"if", "for", "while", "try"}

    score = toolset.compare_complexity_to_typical("cyclomatic_complexity", stats["cyclomatic_complexity"])
    assert 0.0 <= score <= 1.0


def test_structural_toolset_does_not_crash_on_unparseable_code():
    views = split_code("this is not { valid python code :::", language="python")
    toolset = StructuralToolset(views, language="python")
    stats = toolset.get_ast_stats()  # must not raise
    assert "max_nesting_depth" in stats
