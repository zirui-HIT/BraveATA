from .base import PackerBase
from ..data_item import DataItem

from typing import Tuple


class PackerNLI(PackerBase):
    prompt = """
Given a premise and a hypothesis, classify their relationship.
Think it step by step.

Definitions:
- entailment: the hypothesis is definitely true based on the premise
- contradiction: the hypothesis is definitely false based on the premise
- neutral: the premise does not provide enough information to determine whether the hypothesis is true or false

Premise: {context}
Hypothesis:
\"\"\"
{hypothesis}
\"\"\"

Return your answer with the format "The answer is: <your answer>".
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            context=data.get_context(),
            hypothesis=data.get_hypothesis()
        ), ""
