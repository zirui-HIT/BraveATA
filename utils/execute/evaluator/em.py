from .base import EvaluatorBase

from typing import List, Union


class EvaluatorEM(EvaluatorBase):
    def evaluate(self, pred: str, gold: Union[List[str], str]) -> bool:
        if isinstance(gold, str):
            gold = [gold]

        for g in gold:
            if pred.strip().lower() == g.strip().lower():
                return True
        return False
