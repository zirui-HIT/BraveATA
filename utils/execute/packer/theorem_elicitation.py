from ..data_item import DataItem
from .base import PackerBase

from typing import Tuple


class PackerTheoremElicitation(PackerBase):
    prompt = """
You are given a `core_claim` about a theoretical result in machine learning theory or transformer theory.

Your task is to write the corresponding `informal_theorem` in clear, precise, textbook-style mathematical English.

The output must be a theorem statement, not an explanation, not a proof sketch, and not a paraphrase of the claim.

Requirements:

1. Expand the core claim into a self-contained theorem statement.
   - Introduce the mathematical setting explicitly.
   - Define the relevant objects, variables, model class, data distribution, optimization process, or architecture.
   - State the assumptions needed for the claim to make sense.
   - End with the exact conclusion implied by the claim.

2. Use theorem-style structure.
   - Prefer a structure like:
     "Consider ..."
     "Let ..."
     "Assume ..."
     "Then ..."
   - The result should read like a theorem from a paper or textbook.

3. Be faithful to the core claim.
   - Do not strengthen the claim.
   - Do not add conclusions that are not supported by the claim.
   - Do not introduce unnecessary novelty.
   - Preserve the original meaning, scope, and direction of the result.

4. Make the statement mathematically precise.
   - Replace vague phrases with concrete mathematical conditions whenever they are naturally implied by the claim.
   - If the claim refers to asymptotic behavior, sample complexity, convergence, approximation, error bounds, expressivity, memorization, robustness, or generalization, state that precisely.
   - If the claim involves probability, optimization, or existence, use standard theorem language such as:
     "there exists", "with probability at least", "for every", "for sufficiently large", "converges to", "is bounded by", etc.

5. Include only the ingredients needed for the theorem.
   - Include assumptions, notation, and conclusion.
   - Do not include proof ideas, intuition, motivation, or commentary.
   - Do not mention the source paper, theorem number, authors, or phrases like "this paper shows".

6. Make it self-contained.
   - The theorem should be understandable on its own without reading the core claim.
   - Any symbol introduced in the theorem should be defined before it is used.

7. Style constraints.
   - Write in one coherent paragraph unless multiple paragraphs are clearly necessary.
   - Use formal but natural mathematical English.
   - Keep notation readable and standard.
   - Avoid bullet points.
   - Avoid excessive verbosity, but include all essential assumptions and the full conclusion.

8. When details are missing from the core claim:
   - Infer the minimal standard setup needed to state the theorem cleanly.
   - Stay conservative: prefer generic but valid theorem phrasing over unsupported specificity.
   - If an exact constant, rate, or formula is not recoverable from the claim, state the result in a mathematically correct high-level form instead of inventing details.

Input:
core_claim: {core_claim}

Output:
Return only the informal theorem text in the format "The answer is: <your answer>".
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            core_claim=data.get_core_claim()
        ), ""
