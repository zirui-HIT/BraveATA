import re
import json

from .f1 import EvaluatorF1
from ..data_item import DataItem
from ...generator import detect_generator

from typing import List, Dict, Any, Optional


def extract_score_json(model_output: str) -> Dict[str, Any]:
    """
    Extract the JSON score payload from model output and return it as a dict.
    Return an empty dict {} if extraction or parsing fails.

    Extraction strategy:
    1. Prefer JSON found inside ```json ... ``` or generic fenced code blocks.
    2. If that fails, scan the full text for the first complete JSON object.
    3. Return only dict results; otherwise return {}.
    """

    def _try_parse_dict(s: str) -> Dict[str, Any]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    if not isinstance(model_output, str) or not model_output.strip():
        return {}

    # 1) Prefer extracting JSON from fenced code blocks.
    code_block_patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in code_block_patterns:
        for candidate in re.findall(pattern, model_output, flags=re.IGNORECASE):
            candidate = candidate.strip()
            parsed = _try_parse_dict(candidate)
            if parsed:
                return parsed

    # 2) Scan the full text for the first parsable complete JSON object.
    text = model_output.strip()
    start_idx: Optional[int] = None
    brace_count = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if start_idx is None:
            if ch == "{":
                start_idx = i
                brace_count = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start_idx:i + 1]
                    parsed = _try_parse_dict(candidate)
                    if parsed:
                        return parsed
                    # This slice is not valid JSON; continue searching.
                    start_idx = None
                    brace_count = 0
                    in_string = False
                    escape = False

    return {}


class EvaluatorLLMRubric(EvaluatorF1):
    prompt = """
You are an expert evaluator of informal mathematical statements and arguments.

Your job is to compare a candidate informal mathematical item against the reference informal mathematical item from an annotated benchmark example, and score the candidate on four dimensions:
1. logical_validity
2. completeness
3. correctness
4. clarity

The item may be either:
- an informal theorem statement, or
- an informal proof.

You must first determine which kind of item is being evaluated, unless the input explicitly specifies it.

You must be strict, but fair:
- Do NOT require identical wording.
- Accept paraphrases, reordered presentation, different notation, and different proof organization if the mathematical meaning is preserved.
- Do penalize missing assumptions, changed quantifiers, altered claim strength, shifted scope, incorrect conclusions, contradictions, unsupported steps, invalid inferences, omitted key cases, circular reasoning, or vague/imprecise wording.
- A theorem/proof can be clear but still incorrect.
- A theorem/proof can be logically valid as a standalone piece of writing but still incorrect relative to the reference.
- Score each dimension independently on a continuous scale in [0, 1].

Scoring guidance:
- 1.0 = essentially perfect on that dimension
- 0.8 = very strong, only minor issues
- 0.6 = mostly good, but noticeable issues
- 0.4 = major weaknesses
- 0.2 = severe problems
- 0.0 = fundamentally broken

General evaluation rules:
- Compare semantic content, not surface form.
- Do not penalize harmless stylistic changes.
- Do penalize loss of essential mathematical meaning.
- If the candidate is weaker or stronger than the reference in a mathematically material way, correctness must decrease.
- If the candidate omits technical assumptions or key proof steps but the main idea is still roughly captured, completeness should decrease more than clarity.
- If the candidate is fluent but mathematically wrong, correctness should be low.
- If the candidate is mathematically faithful but awkwardly written, clarity can be low while correctness remains high.

Dimension definitions:

1. logical_validity
Whether the candidate is internally coherent as a mathematical item.

- For an informal theorem:
  Check whether assumptions, objects, notation, and conclusion fit together without contradiction, malformed quantification, undefined terms, category errors, or broken logical structure.

- For an informal proof:
  Check whether the proof is internally coherent as an argument: steps should connect sensibly, the proof should target the intended claim, and there should be no blatant contradictions, non sequiturs, circularity, or misuse of assumptions/notation.

2. completeness
Whether the candidate covers the essential ingredients of the reference.

- For an informal theorem:
  Check whether it includes the main assumptions, important conditions, relevant objects, key regimes/cases, and the core conclusion.

- For an informal proof:
  Check whether it includes the essential strategy and key intermediate steps of the reference proof, including important lemmas, constructions, reductions, case splits, boundary conditions, and how the conclusion is obtained from the assumptions.

3. correctness
Whether the candidate faithfully matches the mathematical content of the reference.

- For an informal theorem:
  Penalize incorrect assumptions, incorrect dependencies, wrong bounds, wrong direction of implication, overclaiming, underclaiming, changed quantifiers, materially different hypotheses, or a materially different conclusion.

- For an informal proof:
  Penalize invalid inferences, unsupported claims, use of unjustified lemmas, missing crucial conditions, proving only a weaker claim, proving a different claim, incorrect algebra/logic, or reasoning that does not actually establish the reference result. Also penalize proofs that are internally plausible but do not match the reference argument in mathematically essential ways.

4. clarity
Whether the candidate is easy to read and interpret.

- For either theorem or proof:
  Reward precise wording, explicit assumptions, unambiguous logical flow, and mathematically natural phrasing.
  Penalize vagueness, ambiguity, confusing notation, missing connective structure, or poorly organized sentences.

Important item-type-specific rules:

If the item is an informal theorem:
- Focus on hypotheses, quantifiers, objects, and conclusion.
- Do not require proof details.
- A theorem may be logically valid even if it states the wrong result relative to the reference.

If the item is an informal proof:
- Focus on whether the proof actually supports the intended theorem/claim.
- Accept different proof styles, proof order, or different but valid proof strategies, as long as they establish essentially the same claim.
- Do not penalize omission of low-level routine algebraic details if the main argument is mathematically sound and the omitted details are standard.
- Do penalize omission of nontrivial steps that are necessary for the argument to go through.
- If the candidate proof is only a proof sketch while the reference is a full informal proof, reduce completeness but not necessarily clarity.
- If the candidate reaches the correct conclusion for the wrong reasons, correctness must be low.

Input:
item_type:
{item_type}

reference_item:
```
{reference_item}
```

candidate_item:
```
{candidate_item}
```

Instructions for item_type:
- If item_type is "theorem", evaluate as an informal theorem statement.
- If item_type is "proof", evaluate as an informal proof.
- If item_type is missing, empty, or "auto", infer the type from the reference and candidate. If they mismatch in type, treat that as a major issue for completeness and correctness.

Output:
Return JSON only, with no extra text, in exactly this format:
```json
{{
  "logical_validity": {{
    "score": <float in [0,1]>,
    "reason": "<brief reason>"
  }},
  "completeness": {{
    "score": <float in [0,1]>,
    "reason": "<brief reason>"
  }},
  "correctness": {{
    "score": <float in [0,1]>,
    "reason": "<brief reason>"
  }},
  "clarity": {{
    "score": <float in [0,1]>,
    "reason": "<brief reason>"
  }}
}}
```
""".strip()

    def __init__(
        self,
        llm_name_or_path: str = "gpt-5.4",
        config_path: str = "./generate/config/gpt-5.4.json"
    ):
        self.generator = detect_generator(llm_name_or_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            self.config['n'] = 1

        super().__init__()

    def evaluate_multi(self, data: List[DataItem], **kwargs) -> List[Dict[str, Any]]:
        prompts = [{
            "instruction": self.prompt.format(
                item_type=kwargs['answer_key'],
                reference_item=d.data['natural_language'][kwargs['answer_key']],
                candidate_item=p['answer']
            ),
            "response": ""
        } for d in data for p in d.data['prediction'][kwargs['task']]]
        results = self.generator.generate(prompts, self.config)
        result_idx = 0
        data_updated = []
        for d in data:
            for p in d.data['prediction'][kwargs['task']]:
                r = results[result_idx]
                result_idx += 1
                p['evaluation'] = {
                    "f1": self.evaluate(p['answer'], [d.data['natural_language'][kwargs['answer_key']]]),
                    "origin": r[0][0]
                }
                p['evaluation'].update(extract_score_json(r[0][0]))
                data_updated.append(d.data)
        print(json.dumps(
            data_updated[0]['prediction'][kwargs['task']][0], ensure_ascii=False, indent=4))
        return data_updated
