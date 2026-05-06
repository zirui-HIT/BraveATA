from .base import PackerBase
from ..data_item import DataItem

from typing import Tuple


class PackerClaimGeneration(PackerBase):
    prompt = """
You are a research assistant for mechanistic interpretability.

Your job is to propose exactly one novel core claim about an internal mechanism of a language model, which must can be mathematically proved.

A good core claim must:
1. describe an internal mechanism rather than an external behavior,
2. mention at least one concrete internal object or representation,
3. assert a causal relationship,
4. be falsifiable by a plausible intervention, ablation, or counterfactual test,
5. be self-contained and concise,
6. avoid citing papers, authors, benchmarks, or prior work.

Do not provide explanations, caveats, examples, or multiple alternatives.
Return your answer with the format "The answer is: <your answer>".
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt, ""
