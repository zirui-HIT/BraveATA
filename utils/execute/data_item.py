from typing import Dict, List, Any


class DataItem(object):
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_context(self) -> str:
        return self.data['context']

    def get_question(self) -> str:
        return self.data['question']

    def get_answer(self) -> str:
        for part in ['answer', 'gold_answers']:
            if part in self.data:
                return self.data[part]
        raise Exception("No answer found in data")

    def get_hypothesis(self) -> str:
        if 'claim' in self.data:
            return self.data['claim']
        pred_answer = str(self.data['pred_answers']) if len(
            self.data['pred_answers']) > 1 else self.data['pred_answers'][0]
        return f"The answer of \"{self.data['question']}\" is \"{pred_answer}\""

    def get_rationale(self) -> str:
        if 'rationale' in self.data:
            rationale = self.data['rationale']
        else:
            rationale = self.data['prediction']['rationale']
        if "<think>" in rationale:
            rationale = rationale.split("</think>")[-1].strip()
        if rationale.startswith("<think>"):
            rationale = rationale[len("<think>"):].strip()
        return rationale

    def get_core_claim(self) -> str:
        return self.data['natural_language']['core_claim']

    def get_src_header(self) -> str:
        return self.data['formal_language']['src_header']

    def get_informal_theorem(self) -> str:
        return self.data['natural_language']['informal_theorem']

    def get_formal_theorem(self) -> str:
        return self.data['formal_language']['formal_theorem']
