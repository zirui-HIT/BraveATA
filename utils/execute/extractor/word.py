from .base import ExtractorBase


class ExtractorWord(ExtractorBase):
    def extract(self, text: str) -> str:
        try:
            answer = text.split('answer is')[-1].strip(': \n')
            answer = answer.split()[0].strip()
            answer = answer.strip(' :*.$\",')
        except:
            answer = text
        return answer
