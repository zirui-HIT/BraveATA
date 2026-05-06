from .base import ExtractorBase


class ExtractorCode(ExtractorBase):
    def extract(self, text: str) -> str:
        try:
            if "```Lean" in text:
                answer = text.split('```Lean')[-1]
                answer = answer.split('```')[0].strip()
            else:
                answer = text.split('```')[0].strip()
        except:
            answer = text
        return answer
