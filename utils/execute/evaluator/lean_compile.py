import os
import re
import shutil
import tempfile
import textwrap
import subprocess
import uuid

from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Any

from .beq_plus import clean_lean
from .base import EvaluatorBase
from ..data_item import DataItem
from tool.check_lean_declaration_syntax import *


def _default_project_path() -> str:
    return str(Path(__file__).resolve().parents[3] / "LeanTest")


FORBIDDEN_WORDS = ["sorry", "admit", "assume", "axiom", "postulate"]


def combine_theorem_and_proof(theorem: str, proof: str, indent: int = 4) -> str:
    """
    Combine a theorem header and proof into a Lean declaration while keeping
    indentation stable.

    Rules:
    1. If theorem already ends with :=, strip it first and emit a single normalized one.
    2. If proof starts with by, emit it as:
           theorem ... := by
             ...
    3. If proof is a single-line term (for example rfl / exact h), place it after :=.
    4. If proof is a multi-line term (for example calc / fun / match), indent it and place it on the next line after :=.

    Notes:
    - This function only handles formatting and indentation normalization.
    - It does not replace Lean parser validation of the proof itself.
    """
    def _normalize_newlines(s: str) -> str:
        return (s or "").replace("\r\n", "\n").replace("\r", "\n")

    def _normalize_whitespace(s: str, tabsize: int = 2) -> str:
        # Normalize CRLF to LF and expand tabs to spaces to reduce Lean indentation ambiguity.
        return _normalize_newlines(s).expandtabs(tabsize)

    def _dedent_block(s: str) -> str:
        """
        Remove shared leading indentation from a multi-line proof.
        For a single line, also trim surrounding whitespace.
        """
        s = _normalize_whitespace(s).strip("\n")
        if not s:
            return ""

        s = textwrap.dedent(s)

        # Single-line proof: trim surrounding whitespace, e.g. "  rfl  " -> "rfl".
        if "\n" not in s:
            return s.strip()

        # Multi-line proof: keep internal indentation and trim outer blank lines.
        return s.strip("\n")

    theorem = _normalize_whitespace(theorem).strip()
    proof = _dedent_block(proof)

    if not theorem:
        return ""
    if not proof:
        return ""
    if indent < 0:
        return ""

    # Strip an existing trailing := from the theorem header.
    theorem = re.sub(r"\s*:=\s*$", "", theorem)
    pad = " " * indent

    lines = proof.splitlines()
    first = lines[0].strip()

    # Case 1: proof looks like:
    # by
    #   ...
    if first == "by":
        body = _dedent_block("\n".join(lines[1:]))
        if not body:
            return f"{theorem} := by\n"
        return f"{theorem} := by\n{textwrap.indent(body, pad)}\n"

    # Case 2: proof looks like:
    # by trivial
    # or
    # by exact h
    if first.startswith("by "):
        first_body = first[3:].lstrip()
        rest = "\n".join(lines[1:])
        body = first_body if not rest else first_body + "\n" + rest
        body = _dedent_block(body)
        return f"{theorem} := by\n{textwrap.indent(body, pad)}\n"

    # Case 3: proof is a single-line term, for example:
    # rfl
    # exact h
    if "\n" not in proof:
        return f"{theorem} := {proof}\n"

    # Case 4: proof is a multi-line term, for example:
    # calc
    #   ...
    # fun x =>
    #   ...
    return f"{theorem} :=\n{textwrap.indent(proof, pad)}\n"


def check_lean4_proofs_verbose(
    data: List[Tuple[str, str, str]],
    project_path: str,
    timeout_sec: int = 30,
    keep_failed_files: bool = True,
    include_warnings: bool = True,
    include_infos: bool = False,
) -> List[str]:
    """
    Input:
        data: List[Tuple[source_header, theorem, proof]]
            - source_header: Lean file header, for example imports / open / namespace / set_option.
            - theorem: The theorem declaration header, without the proof.
            - proof: The proof body (either `by ...` or a regular term proof).
        project_path: Root directory of the Lean4 project.

    Output:
        List[str]
        - Successful compilation: "pass"
        - Failed compilation: a more detailed Lean error message
    """
    root = validate_project_root(project_path)
    ensure_lake_available()

    debug_dir = root / ".lean_syntax_debug"
    if keep_failed_files:
        debug_dir.mkdir(exist_ok=True)

    results: List[str] = []

    with tempfile.TemporaryDirectory(prefix="lean_syntax_check_", dir=root) as tmpdir:
        tmpdir_path = Path(tmpdir)

        for idx, (header, theorem, proof) in tqdm(
            enumerate(data),
            desc="Checking Lean proofs",
            total=len(data),
        ):
            uid = uuid.uuid4().hex[:8]
            file_name = f"Check_{idx}_{uid}.lean"
            file_path = tmpdir_path / file_name

            decl = combine_theorem_and_proof(theorem, proof)
            if not decl:
                results.append("combine error")
                continue

            # Reuse the existing debug-source composition logic.
            source, meta = compose_debug_lean_source(header, decl)
            file_path.write_text(source, encoding="utf-8")

            if idx == 0:
                print(source)
            continue_flag = False
            for w in FORBIDDEN_WORDS:
                if w in source:
                    results.append(f"forbidden word '{w}' found")
                    continue_flag = True
                    break
            if continue_flag:
                continue

            rel_file_path = file_path.relative_to(root)

            cmd = [
                "lake",
                "lean",
                str(rel_file_path),
                "--",
                "--json",
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                results.append(f"timeout after {timeout_sec}s")
                continue
            except Exception as e:
                results.append(
                    f"failed to invoke Lean: {type(e).__name__}: {e}")
                continue

            combined_output = join_outputs(proc.stdout, proc.stderr)
            messages = extract_all_lean_messages(combined_output)

            # Look for errors first: only an error or non-zero exit code counts as failure.
            errors = []
            for m in messages:
                sev = str(m.get("severity", "")).lower()
                if sev == "error":
                    errors.append(m)

            # Successful compilation: return pass directly.
            if proc.returncode == 0 and not errors:
                results.append("pass")
                continue

            # On failure, decide whether warnings and info messages should also be shown.
            interesting = []
            for m in messages:
                sev = str(m.get("severity", "")).lower()
                if sev == "error":
                    interesting.append(m)
                elif sev == "warning" and include_warnings:
                    interesting.append(m)
                elif sev == "information" and include_infos:
                    interesting.append(m)

            if not interesting:
                fallback = extract_plain_error_text(proc.stdout, proc.stderr)
                results.append(
                    fallback or f"Lean exited with code {proc.returncode}")
                continue

            saved_file = None
            if keep_failed_files and errors:
                saved_file = debug_dir / file_name
                saved_file.write_text(source, encoding="utf-8")

            rendered = render_detailed_messages(
                messages=interesting,
                source=source,
                display_path=str(
                    saved_file if saved_file is not None else rel_file_path),
                header_line_count=meta["header_line_count"],
                prelude_line_count=meta["prelude_line_count"],
            )
            results.append(rendered)

    return results


class EvaluatorLeanCompile(EvaluatorBase):
    def __init__(
        self,
        time_out=1800,
        project_path=_default_project_path()
    ):
        self.time_out = time_out
        self.project_path = project_path
        super().__init__()

    def evaluate_multi(self, origin_data: List[DataItem], **kwargs) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = [d.data for d in origin_data]
        leans = [(d['formal_language']['src_header'], clean_lean(d['formal_language']['formal_theorem']),
                  p['answer']) for d in data for p in d['prediction']['theorem_proving_formal']]
        results = check_lean4_proofs_verbose(
            data=leans,
            project_path=self.project_path,
            timeout_sec=self.time_out
        )
        result_idx = 0
        result_data = []
        for d in tqdm(data):
            for p in d['prediction']['theorem_proving_formal']:
                r = results[result_idx]
                result_idx += 1
                p['evaluation'] = {
                    # "info": r,
                    "pass": bool(r == "pass")
                }
            result_data.append(d)

        return result_data
