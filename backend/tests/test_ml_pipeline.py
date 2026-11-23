"""
Tests 1, 3, 4: ModernBERT loading, preprocessing, code splitting.
(Classifier checkpoint loading -- test 2 -- requires a real
best_4class_classifier.pt file and is covered by
test_classifier_checkpoint.py, skipped automatically when the file
is absent.)
"""

import os

import pytest

from app.ml.cleaner import clean_code
from app.ml.code_splitter import split_code


def test_clean_code_normalizes_line_endings_and_blank_lines():
    raw = "def f():\r\n    pass\r\n\r\n\r\n\r\nprint(1)"
    cleaned = clean_code(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_code_handles_empty_input():
    assert clean_code("") == ""
    assert clean_code("   \n  ") == ""
    assert clean_code(None) == ""


def test_split_code_python_extracts_function_and_comment():
    source = '''
def add(a, b):
    # add two numbers
    return a + b
'''
    views = split_code(source, language="python")
    assert "def add" in views.sub_functions
    assert "add two numbers" in views.comments_view
    assert views.parse_status == "ast_python"


def test_split_code_empty_input_uses_placeholder():
    views = split_code("", language="python")
    assert views.main_code == "[NONE]"
    assert views.sub_functions == "[NONE]"
    assert views.comments_view == "[NONE]"


def test_split_code_non_python_falls_back_to_regex():
    source = '''
// increments x
function increment(x) {
    return x + 1;
}
'''
    views = split_code(source, language="javascript")
    assert views.parse_status == "regex_fallback"
    assert "increment" in views.sub_functions
    assert "increments x" in views.comments_view


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("MODERNBERT_MODEL_PATH", "")),
    reason="MODERNBERT_MODEL_PATH not set / file not present -- place the real "
           "T2 checkpoint to exercise this test.",
)
def test_modernbert_encoder_loads_and_is_frozen():
    from app.ml.modernbert_encoder import ModernBertEncoder

    encoder = ModernBertEncoder()
    assert encoder.is_frozen()
    last_hidden_state, attention_mask = encoder.encode(["def f(x): return x"])
    assert last_hidden_state.shape[-1] == encoder.hidden_size
