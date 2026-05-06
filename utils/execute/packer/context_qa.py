from ..data_item import DataItem
from .base import PackerBase

from typing import Tuple


class PackerContextQA(PackerBase):
    prompt = """
You are a careful question-answering assistant.

Your task is to answer the question strictly based on the given context.
Think it step by step.

Rules:
1. Use only the information supported by the context.
2. Do not rely on outside knowledge unless the context explicitly supports it.
3. Do not guess, speculate, or add extra details.
4. Output the final answer with the format "The answer is: <your answer>"   

---

Context:
{context}

Question:
{question}
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            context=data.get_context(),
            question=data.get_question()
        ), ""
