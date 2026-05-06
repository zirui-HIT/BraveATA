from .f1 import EvaluatorF1

from typing import List, Union


class EvaluatorInclude(EvaluatorF1):
    def evaluate(self, pred: str, gold: List[str]) -> float:
        assert isinstance(pred, str) and isinstance(gold, list)
        for g in gold:
            if g.lower() in pred.lower():
                return 1.0
        return super().evaluate(pred, gold)
