from .base import ExtractorBase


class ExtractorSentence(ExtractorBase):
    def extract(self, text: str) -> str:
        try:
            answer = text.split('answer is:')[-1].strip()
            # answer = answer.split('\n\n')[0]
            answer = answer.strip(' :*.$\",\n')
        except:
            answer = text
        return answer
