from ..data_item import DataItem

from typing import Dict, Any


class EvaluatorBase:
    def evaluate(self, data: DataItem) -> Dict[str, Any]:
        return data.data
