"""Prompts of the evaluation set checks. The documents and questions are Chinese, so are these."""

from __future__ import annotations

PASSAGE_TEMPLATE = "证据 {i}（{label}）：\n<<<\n{text}\n>>>"

CLOSED_BOOK_TEMPLATE = """请凭你自己的知识回答下面关于{subject}的问题。即使不确定也给出你认为最可能的答案，\
不要拒答。只回复简短答案。

问题：{question}"""

# Bump when the pre-review prompt changes; each pre-review records the version it used.
PREREVIEW_PROMPT_VERSION = 1

PREREVIEW_TEMPLATE = """你在审核一个评测集的候选题。评测集基于{subject}的文档，用来测试检索 Agent，\
所以每道题都必须能从文档中得到答案，而不能靠常识答出来。

审核标准：
{criteria}

结论：
- "keep"：完全符合标准，原样保留。
- "modify"：改写问题或答案后可以保留，给出改好的版本。
- "drop"：无法修正。
"modify" 和 "drop" 要给出一个或多个原因，只能从这些 id 中选：{reasons}。

题型：{group}
问题：{question}
标准答案：{answer}
也接受的答案：{aliases}

证据：
{passages}

同一文档上的其他候选题：
{others}

只回复 JSON：
{{"decision": "keep|modify|drop", "reasons": [], "explanation": "一两句话", \
"suggested_question": null, "suggested_answer": null}}"""


def passages_block(passages: list[tuple[str, str]]) -> str:
    """``(label, text)`` pairs as numbered, delimited passages."""
    return "\n\n".join(
        PASSAGE_TEMPLATE.format(i=i, label=label, text=text)
        for i, (label, text) in enumerate(passages, 1)
    )
