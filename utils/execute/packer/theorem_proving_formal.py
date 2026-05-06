from ..data_item import DataItem
from .base import PackerBase
from ..evaluator.beq_plus import clean_lean

from typing import Tuple


class PackerTheoremProvingFormal(PackerBase):
    prompt = """
You are given two inputs:

1. `src_header`: the exact Lean 4 import header.
2. `formal_theorem`: a Lean theorem declaration whose statement must be preserved exactly.

Your task is to generate the corresponding Lean 4.29.0 formal proof.

Goal:
Complete the proof of the given theorem in valid Lean 4.29.0 syntax, using the environment provided by `src_header`, while preserving the theorem statement exactly.

Important evaluation detail:
Your output will be extracted verbatim and concatenated directly after `src_header` and `formal_theorem` for evaluation. Therefore:

* Do not repeat or restate `src_header`.
* Do not repeat or restate `formal_theorem`.
* Output only the formal proof that should come after `formal_theorem`.
* Do not include markdown fences, labels, explanations, or any surrounding text.

Instructions:

1. Produce a real Lean proof

* The proof must be valid Lean 4.29.0 code.
* Do not use `sorry`, `admit`, `axiom`, `by_contra!` if unsupported, or any placeholder.
* Prefer concise and robust proofs.
* Use standard Mathlib tactics and lemmas when appropriate.
* When possible, use tactics such as:
  `simp`, `simpa`, `aesop`, `rfl`, `rw`, `exact`, `refine`, `constructor`, `rcases`, `obtain`, `have`, `calc`, `linarith`, `omega`, `ring`, `nlinarith`, `ext`
  only when they are appropriate and valid.
* If a term-style proof is cleaner, use a term-style proof.

2. Proof engineering requirements

* Respect Lean’s typing strictly.
* Use only identifiers that are in scope from the theorem statement or available from the imported libraries.
* If intermediate lemmas are needed, define them locally inside the proof using `have`.
* Do not introduce top-level helper lemmas or extra declarations unless absolutely necessary.
* Keep the proof as short as possible, but prioritize correctness.

3. When the theorem is difficult

* First try to prove it directly from the hypotheses in the theorem.
* Exploit definitional equalities, `simp` lemmas, extensionality, and rewriting opportunities before attempting complicated tactics.
* If the statement is effectively a wrapper around an abstract predicate or hypothesis, use the corresponding assumption directly.
* If the theorem is an equivalence, conjunction, or existential statement, match the proof structure carefully with the goal.
* Do not hallucinate unavailable lemmas.

4. Output restrictions

* Output only Lean code.
* Output only the proof content that is meant to be appended after `formal_theorem`.
* Do not output `src_header`.
* Do not output `formal_theorem`.
* Do not use markdown fences.
* Do not explain the proof.
* Do not output anything except the proof itself.

Input:
src_header:
```
{src_header}
```
formal_theorem:
```
{formal_theorem}
```

Output:
Return the Lean 4.29.0 formal proof in the format "The answer is:\n```Lean\n<your answer>\n```" without repeating src_header and formal_theorem.
""".strip()
    output_prompt = """
```Lean
{src_header}

{formal_theorem}
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            src_header=data.get_src_header(),
            formal_theorem=clean_lean(data.get_formal_theorem())
        ), self.output_prompt.format(
            src_header=data.get_src_header(),
            formal_theorem=clean_lean(data.get_formal_theorem())
        )
