from .base import PackerBase
from ..data_item import DataItem

from typing import Tuple


class PackerSummary(PackerBase):
    prompt = """
Please summarize the following text clearly and concisely.

Requirements:
- Capture the main ideas and key supporting points.
- Preserve the original meaning and tone as much as possible.
- Avoid unnecessary details, repetition, and examples unless they are essential.
- Write in plain, natural English.
- Keep the summary within [desired length, e.g. 100 words / 3 bullet points / one paragraph].
- If the text contains action items, decisions, or important conclusions, highlight them separately.
- Format your answer with "The answer is: <your answer>"

Text:
{text}
""".strip()

    def pack(self, data: DataItem) -> Tuple[str, str]:
        return self.prompt.format(text=data.get_context()), ""
