import re
import os
import json
import tempfile

from pathlib import Path
from .base import EvaluatorBase
from ..data_item import DataItem
from ...BEq_plus.beq_plus import *

from typing import List, Dict, Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_project_path() -> str:
    return str(_repo_root() / "LeanTest")


def _default_repl_path() -> str:
    return str(
        _repo_root()
        / "utils"
        / "BEq_plus"
        / "lean_interact"
        / "cache"
        / "BEq_plus"
        / "repl"
    )


def dump_json(data: List[Dict[str, Any]], dump_file: str):
    dir_name = os.path.dirname(os.path.abspath(dump_file)) or "."
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=dir_name,   # Keep the temp file in the target directory for atomic replacement.
            delete=False
        ) as f:
            tmp_file = f.name
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # Try to ensure the data reaches disk.
        os.replace(tmp_file, dump_file)  # Overwrite only after the write succeeds.
    except Exception:
        # Delete the temp file on failure and keep the original file unchanged.
        if tmp_file and os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise


def clean_lean(lean: str) -> str:
    s = lean.rstrip()
    pattern = re.compile(r'(^|\s+)(by|sorry)\s*$')
    while True:
        m = pattern.search(s)
        if not m:
            break
        s = s[:m.start()].rstrip()
    return s


class EvaluatorBEqPlus(EvaluatorBase):
    def __init__(
        self,
        time_out: int = 1800,
        project_path: str = _default_project_path(),
        repl_path: str = _default_repl_path()
    ):
        self.repl_config = LeanREPLConfig(
            project=LocalProject(
                directory=project_path),
            local_repl_path=repl_path,
            verbose=True
        )
        self.server = AutoLeanServer(config=self.repl_config)
        self.metric = beq_plus
        self.time_out = time_out
        super().__init__()

    def evaluate_multi(self, origin_data: List[DataItem], **kwargs) -> List[Dict[str, Any]]:
        console.print(f"{self.metric.__name__} metric on examples:")
        data = [d.data for d in origin_data]
        for d in data:
            for p in d['prediction']['autoformalization']:
                pred = clean_lean(p['answer'])
                gold = clean_lean(d['formal_language']['formal_theorem'])
                src_header = d['formal_language']['src_header']
                if 'evaluation' in p:
                    continue

                try:
                    equivalent = self.metric(
                        pred,
                        gold,
                        src_header,
                        self.server,
                        timeout_per_proof=self.time_out,
                        verbose=True,
                    )
                    p['evaluation'] = {
                        "equivalent": equivalent
                    }
                except Exception as e:
                    console.print(
                        f"Error evaluating example with src_header={src_header}: {e}")
                    p['evaluation'] = {
                        "equivalent": 0.0
                    }
                if "dump_file" in kwargs:
                    dump_json(data, kwargs["dump_file"])
        return data
