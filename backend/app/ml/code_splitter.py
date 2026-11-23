"""
ml/code_splitter.py
=====================
REUSED, unmodified in behavior from the existing project's
src/preprocessing/code_splitter.py. Produces the three views
(main_code, sub_functions, comments_view) that both the ML branch
and the two agents' toolsets consume.
"""

from dataclasses import dataclass

from app.config import EMPTY_VIEW_PLACEHOLDER, FUNCTION_JOIN_SEPARATOR
from app.ml.cleaner import clean_code
from app.ml.code_parser import parse_source


@dataclass
class CodeViews:
    main_code: str
    sub_functions: str
    comments_view: str
    parse_status: str
    functions_list: list  # raw list of extracted function/method source blocks
    comments_list: list   # raw list of extracted comment strings


def split_code(source_code: str, language: str = None) -> CodeViews:
    cleaned = clean_code(source_code)

    if not cleaned:
        return CodeViews(
            main_code=EMPTY_VIEW_PLACEHOLDER,
            sub_functions=EMPTY_VIEW_PLACEHOLDER,
            comments_view=EMPTY_VIEW_PLACEHOLDER,
            parse_status="unsupported",
            functions_list=[],
            comments_list=[],
        )

    parse_result = parse_source(cleaned, language=language)

    sub_functions = (
        FUNCTION_JOIN_SEPARATOR.join(parse_result.functions)
        if parse_result.functions
        else EMPTY_VIEW_PLACEHOLDER
    )
    comments_view = (
        "\n".join(parse_result.comments)
        if parse_result.comments
        else EMPTY_VIEW_PLACEHOLDER
    )

    return CodeViews(
        main_code=cleaned,
        sub_functions=sub_functions,
        comments_view=comments_view,
        parse_status=parse_result.status,
        functions_list=list(parse_result.functions),
        comments_list=list(parse_result.comments),
    )
