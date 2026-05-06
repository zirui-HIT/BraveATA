from .base import EvaluatorBase

from typing import List
from collections import Counter
from nltk.tokenize import word_tokenize


class EvaluatorF1(EvaluatorBase):
    def evaluate(self, pred: str, gold: List[str]) -> float:
        assert isinstance(pred, str) and isinstance(gold, list)

        def _single_f1(p: str, g: str) -> float:
            p_tokens = word_tokenize(p.strip().lower())
            g_tokens = word_tokenize(g.strip().lower())

            if not p_tokens and not g_tokens:
                return 1.0
            if not p_tokens or not g_tokens:
                return 0.0

            common = Counter(p_tokens) & Counter(g_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = num_same / len(p_tokens)
            recall = num_same / len(g_tokens)
            return 2 * precision * recall / (precision + recall)

        if not isinstance(gold, list):
            gold = [gold]
        if not gold:
            return 0.0

        return max(_single_f1(pred, g) for g in gold)
