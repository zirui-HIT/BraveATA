from ..data_item import DataItem

from typing import Tuple


class PackerBase:
    def pack(self, data: DataItem) -> Tuple[str, str]:
        return "", ""

    def pack_with_rationale(self, data: DataItem) -> Tuple[str, str]:
        ins, res = self.pack(data)
        ins = f"{ins}\n\nHint:\n\"\"\"\n{data.get_rationale()}\n\"\"\""
        return ins, res
