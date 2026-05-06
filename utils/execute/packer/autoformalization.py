from ..data_item import DataItem
from .base import PackerBase

from typing import Tuple


class PackerAutoformalization(PackerBase):
    prompt = """
You are given two inputs:

1. `src_header`: a Lean 4 import header provided for context only. Do not reproduce it in the output.
2. `informal_theorem`: a textbook-style informal theorem statement in mathematical English.

Your task is to generate a Lean 4 `formal_theorem` that matches the informal theorem as faithfully as possible while remaining syntactically valid and conservatively formalized.

Important objective:
Produce a theorem statement skeleton that is likely to pass Lean 4 syntax checking, even if full semantic formalization would be difficult. Prefer safe abstraction over risky over-commitment.

Instructions:

1. Output target
- Generate only the Lean 4 theorem code.
- The output must consist of:
  a. exactly one Lean theorem
  b. a proof stub ending with `:= by` followed by `sorry`.
- Do not output explanations, comments, JSON, markdown fences, or any text outside the Lean code.
- Do not repeat or copy `src_header` in the output.

2. Use the header only as context
- Use `src_header` only to infer the available Lean environment, notation style, and imported libraries.
- Do not add extra imports unless they are absolutely necessary and already implied by the header style.
- Do not remove imports from `src_header`, but also do not reproduce them in the output.

3. Formalization strategy
- Translate the informal theorem into a single Lean theorem statement.
- Prioritize syntactic validity and structural faithfulness over deep formal completeness.
- If the informal theorem contains concepts that would be expensive or ambiguous to formalize, introduce them as abstract parameters, predicates, functions, relations, sets, maps, operators, or typeclasses.
- When a notion is mentioned but not fully specified, represent it abstractly instead of inventing a brittle definition.

4. Naming and structure
- Use a theorem name that is specific and dataset-style if possible.
- Introduce all mathematical objects explicitly as theorem parameters.
- Add hypotheses for all assumptions in the informal theorem.
- State the conclusion in the theorem return type after `:`.
- Use Lean-friendly dependent typing only when needed; otherwise prefer simple explicit parameters.

5. Conservative abstraction rules
- If the theorem references a model, algorithm, distribution, loss, optimizer, architecture, or property that is not already formalized in standard Lean math libraries, encode it as an abstract constant or predicate.
- If the theorem refers to “is trained by…”, “is generated as…”, “is optimal”, “is a global minimizer”, “has generalization error…”, etc., model these with abstract predicates or functions rather than trying to define the full machinery.
- If the theorem involves asymptotic notation, probability bounds, or big-O style claims, represent them with abstract predicates/functions if necessary.
- If the theorem uses specialized objects such as transformers, attention, prompts, losses, trajectories, or tasks, introduce abstract types and functions for them.

6. Faithfulness constraints
- Preserve the theorem’s quantifier structure, assumptions, and conclusion as much as possible.
- Do not strengthen the theorem.
- Do not add unsupported claims.
- Do not omit essential assumptions that are needed to state the result.
- If an exact formula appears in the informal theorem, encode it directly when syntactically manageable.
- If an exact formula is too risky, encode the same claim through abstract helper symbols and hypotheses.

7. Proof body
- Do not attempt the proof.
- Always end with:
  `:= by`
  `  sorry`

Input:
src_header:
```
{src_header}
```

informal_theorem:
```
{informal_theorem}
```

Output:
Return only the Lean 4 formalized theorem code with "The answer is:\n```Lean\n<your answer>\n```, without repeating `src_header`.
""".strip()
    output_prompt = """
```Lean
{src_header}
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(
            src_header=data.get_src_header(),
            informal_theorem=data.get_informal_theorem()
        ), self.output_prompt.format(
            src_header=data.get_src_header()
        )
