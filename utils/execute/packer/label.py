from .base import PackerBase
from ..data_item import DataItem

from typing import Tuple


class PackerLabel(PackerBase):
    prompt = """
```md
{prompt}

Answer:
{answer}
```
Explain the reasoning process of the above problem briefly.
"""

    def __init__(self, packer: PackerBase):
        self.packer = packer
        super().__init__()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        instruction, response = self.packer.pack(data)
        return self.prompt.format(
            prompt=instruction,
            answer=data.get_answer()
        ), ""
