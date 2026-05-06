from ..data_item import DataItem
from .base import PackerBase

from typing import Tuple


class PackerTheoremProving(PackerBase):
    prompt = """
You are given two inputs:

1. `informal_theorem`: a textbook-style natural-language theorem statement.
2. `formal_theorem`: a Lean 4 theorem statement that formalizes the same result, possibly using abstract parameters, predicates, functions, and helper assumptions.

Your task is to write the corresponding `informal_proof` in clear, rigorous, textbook-style mathematical English.

Important objective:
Produce a proof sketch that is faithful to both the mathematical content of the informal theorem and the abstraction choices visible in the formal theorem.

Instructions:

1. What to write
- Write an informal proof, not a formal proof.
- The proof should explain the main logical route from assumptions to conclusion.
- It should read like the proof paragraph(s) in a theory paper or textbook.
- Do not output Lean code, pseudocode, bullet points, JSON, or metadata.

2. Faithfulness
- The proof must match the theorem actually stated in `informal_theorem`.
- Use `formal_theorem` as a constraint on what objects, assumptions, and conclusions are present.
- If the Lean theorem introduces abstract predicates or helper functions, interpret them conservatively as formal stand-ins for the mathematical notions in the informal theorem.
- Do not claim steps that would require assumptions not present in either theorem.
- Do not strengthen the result.
- Do not introduce extra conclusions.

3. Preferred proof style
- Organize the proof as a compact but structured proof sketch.
- A typical flow is:
  "Start from ..."
  "First ..."
  "Next ..."
  "Then ..."
  "Finally ..."
- Emphasize the key reductions, invariants, decompositions, comparison arguments, concentration bounds, optimization steps, or continuity/compactness arguments that would make the theorem go through.
- The proof should make clear why the conclusion follows, even if many technical details are omitted.

4. Relation between the two inputs
- Use `informal_theorem` to recover the intended mathematical meaning.
- Use `formal_theorem` to detect:
  - the exact quantifier structure,
  - which assumptions are explicit,
  - which objects are abstracted away,
  - whether the conclusion is existential, asymptotic, probabilistic, optimization-based, geometric, etc.
- If the formal theorem is more abstract than the informal theorem, write the proof in mathematical language, but keep it compatible with the abstract formal statement.
- If the formal theorem encodes the result through helper predicates (for example, `IsGlobalMinimizer`, `Attractive`, `WithProbAtLeast`, `Tendsto`, etc.), explain the proof at the level of those concepts rather than inventing hidden definitions.

5. Level of detail
- Include the main proof ingredients and their order of use.
- You may refer to standard moves such as:
  - reparameterization,
  - reduction to a simpler objective,
  - induction on layers/steps,
  - decomposition into terms,
  - concentration,
  - convexity/smoothness,
  - invariant manifolds,
  - compactness/continuity,
  - spectral decomposition,
  - union bounds,
  - comparison with an optimal predictor,
  - stability/generalization arguments.
- Do not include tedious algebra unless it is central to the proof idea.
- Do not simply restate the theorem.

6. Style constraints
- Write in formal mathematical English.
- Do not mention the source paper, theorem number, authors, or phrases like "the paper proves" or "in the formal theorem above".
- Do not mention Lean, theorem provers, syntax, or formalization artifacts explicitly.
- Do not produce section headers unless truly necessary.
- Prefer one coherent paragraph or a small number of connected paragraphs.

7. Missing details
- When the exact technical proof is not fully recoverable, infer the most standard proof strategy compatible with the theorem.
- Stay conservative and plausible.
- Prefer high-level correctness over fabricated technical detail.
- If a delicate part is clearly handled by an auxiliary estimate/lemma, you may say so in natural mathematical language without inventing a fake named result.

8. Output restriction
- Return only the informal proof text, with no surrounding explanation.

Input:
informal_theorem:
```
{informal_theorem}
```

formal_theorem:
```
{formal_theorem}
```

Output:
Return the natural language informal proof in the format "The answer is: <your answer>".
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            informal_theorem=data.get_informal_theorem(),
            formal_theorem=data.get_formal_theorem()
        ), ""
