import __future__

from .packer import *
from .extractor import *
from .evaluator import *
from .data_item import DataItem

from typing import Tuple, Dict, Any, List


class Executor:
    def __init__(
        self,
        packer: PackerBase,
        extractor: ExtractorBase,
        evaluator: EvaluatorBase,
        task: str
    ):
        self.packer = packer
        self.extractor = extractor
        self.evaluator = evaluator
        self.task = task

    @staticmethod
    def initialize(task: str, label: bool = False, activate_eval: bool = False) -> 'Executor':
        packer_class = {
            "base": PackerBase,
            "context_qa": PackerContextQA,
            "nli": PackerNLI,
            "summary": PackerSummary,
            "claim_generation": PackerClaimGeneration,
            "theorem_elicitation": PackerTheoremElicitation,
            "autoformalization": PackerAutoformalization,
            "theorem_proving": PackerTheoremProving,
            "theorem_proving_formal": PackerTheoremProvingFormal
        }.get(task, PackerBase)
        extractor_class = {
            "base": ExtractorBase,
            "context_qa": ExtractorSentence,
            "nli": ExtractorWord,
            "summary": ExtractorSentence,
            "claim_generation": ExtractorSentence,
            "theorem_elicitation": ExtractorSentence,
            "autoformalization": ExtractorCode,
            "theorem_proving": ExtractorSentence,
            "theorem_proving_formal": ExtractorCode
        }.get(task, ExtractorBase)
        evaluator_class = {
            "base": EvaluatorBase,
            "context_qa": EvaluatorInclude,
            "nli": EvaluatorEM,
            "summary": EvaluatorF1,
            "theorem_elicitation": EvaluatorLLMRubric,
            "theorem_proving": EvaluatorLLMRubric,
            "theorem_proving_formal": EvaluatorLeanCompile,
            "autoformalization": EvaluatorBEqPlus
        }.get(task, EvaluatorBase)
        print(
            f"Initialized Executor with packer {packer_class.__name__}, extractor {extractor_class.__name__}, evaluator {evaluator_class.__name__}")
        return Executor(
            packer=packer_class() if not label else PackerLabel(packer_class()),
            extractor=extractor_class(),
            evaluator=evaluator_class() if activate_eval else EvaluatorBase(),
            task=task
        )

    def pack(self, data: Dict[str, Any]) -> Dict[str, str]:
        result = self.packer.pack(DataItem(data))
        return {
            "instruction": result[0],
            "response": result[1]
        }

    def pack_with_rationale(self, data: Dict[str, Any]) -> Dict[str, str]:
        result = self.packer.pack_with_rationale(DataItem(data))
        return {
            "instruction": result[0],
            "response": result[1]
        }

    def extract(self, response: str) -> str:
        return self.extractor.extract(response)

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data_item = DataItem(data)
        return self.evaluator.evaluate(data_item)

    def evaluate_multi(self, data: List[Dict[str, Any]], dump_file: str | None = None) -> List[Dict[str, Any]]:
        if hasattr(self.evaluator, "evaluate_multi"):
            return self.evaluator.evaluate_multi(
                [DataItem(d) for d in data],
                task=self.task,
                answer_key="informal_theorem" if self.task == "autoformalization" else "informal_proof",
                dump_file=dump_file
            )
        results = []
        for d in data:
            result = self.evaluate(d)
            results.append(result)
        return results
