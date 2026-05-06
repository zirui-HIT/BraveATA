---
pretty_name: "BraveATA"
license: "cc0-1.0"
language:
- en
size_categories:
- n<1K
task_categories:
- text-generation
- question-answering
tags:
- llm-evaluation
- mathematical-reasoning
- theorem-proving
- autoformalization
- lean4
- formal-methods
- benchmark
- croissant
- responsible-ai
---

# BraveATA

**BraveATA** (**BR**oad **A**nd **V**erifiable **E**nd-to-End **A**utomated **T**heoretical **A**nalysis) is a benchmark for evaluating whether large language models can carry out theoretical analysis of LLM-related research claims.

Each instance starts from a concise **core claim** extracted from a theoretical-analysis paper and provides target annotations for intermediate stages of the theoretical-analysis pipeline: an informal theorem, a Lean 4 formal theorem statement, and an informal proof sketch.

This repository is anonymized for double-blind review. It intentionally does not include author names, affiliations, emails, or non-anonymized repository links.

## Dataset Overview

| Property | Value |
|---|---|
| Dataset name | BraveATA |
| Data file | `BraveATA.json` |
| Language | English |
| Size | 178 instances |
| Formal system | Lean 4.29.0 |
| License | CC0 1.0 for the benchmark annotations, metadata, and structure |

BraveATA covers 178 source papers across 11 research directions in theoretical analysis of LLMs. The benchmark is designed to support fine-grained evaluation of the path from a research-level claim to theorem statements and proof-level reasoning.

## Repository Structure

```text
.
├── README.md
├── BraveATA.json          # Main dataset file
└── croissant.json         # Croissant metadata file with RAI metadata
```

## Quick Start

### Create Environment

Create conda environment with
```shell
conda create -n discovery python=3.10 -y
conda activate discovery
pip install -r requirements.txt
```

Create Lean project with
```shell
elan self update
lake +leanprover-community/mathlib4:lean-toolchain new LeanTest math
cd LeanTest
lake exe cache get
lake build
```

### Inference

Inference on BraveATA using local GPU:
```shell
bash generate/script/inference_local.sh
```
or using API:
```shell
bash generate/script/inference_api.sh
```

### Evaluate

Evaluate inference results:
```shell
bash generate/script/evaluate.sh
```

## Data Format

Each item in `BraveATA.json` has three sections: `metadata`, `natural_language`, and `formal_language`.

```json
{
  "metadata": {
    "idx": "WDITMBATIFATHA_482731",
    "conference": "ICLR 2025",
    "source_paper": "What Does It Mean to Be a Transformer? Insights from a Theoretical Hessian Analysis",
    "source_statement": "Theorem 3.1"
  },
  "natural_language": {
    "core_claim": "The outer-product Hessian of a self-attention layer has a fundamentally asymmetric block structure, with the query/key blocks coupling data and weights more richly than the value block.",
    "informal_theorem": "...",
    "informal_proof": "..."
  },
  "formal_language": {
    "src_header": "import Mathlib\n",
    "formal_theorem": "theorem WDITMBATIFATHA_482731 ... := by\n  sorry"
  }
}
```

## Data Fields

| Section | Field | Type | Description |
|---|---|---:|---|
| `metadata` | `idx` | string | Unique sample identifier. |
| `metadata` | `conference` | string | Venue and year of the source paper. |
| `metadata` | `source_paper` | string | Title of the source paper. |
| `metadata` | `source_statement` | string | Identifier of the theorem, lemma, proposition, or other statement used from the source paper. |
| `natural_language` | `core_claim` | string | Concise central claim of the source paper. |
| `natural_language` | `informal_theorem` | string | Self-contained natural-language theorem statement corresponding to the source statement. |
| `natural_language` | `informal_proof` | string | Human-verified proof sketch following the reasoning of the source paper. |
| `formal_language` | `src_header` | string | Lean 4 imports and local declarations needed before the theorem statement. |
| `formal_language` | `formal_theorem` | string | Lean 4 formal theorem statement. Some entries may end with `sorry`; the benchmark focuses on formal statement quality rather than complete proof availability for every instance. |

## Supported Tasks

BraveATA supports evaluation of several intermediate capabilities in automated theoretical analysis.

| Task | Input | Target Output |
|---|---|---|
| Theorem Elicitation (CC2IT) | `core_claim` | `informal_theorem` |
| Autoformalization (IT2FT) | `informal_theorem`, `src_header` | `formal_theorem` |
| Theorem Proving (FT2FP) | `src_header`, `formal_theorem` | formal proof |
| Proof Elicitation (IT2IP) | `informal_theorem` | `informal_proof` |

The benchmark can also be used to analyze bottlenecks across the full claim-to-theorem-to-proof pipeline.

## Evaluation Notes

Recommended evaluation depends on the task:

- **Theorem elicitation and proof elicitation:** use expert review or rubric-based semantic evaluation to assess whether the generated theorem/proof is faithful, complete, and mathematically meaningful.
- **Autoformalization:** check Lean syntax/type correctness and semantic alignment with the informal theorem.
- **Theorem proving:** check whether the generated Lean proof compiles under the intended Lean 4 environment.

The formal theorem statements were developed for Lean 4. Users should report the exact Lean and Mathlib versions used in their experiments.

## Dataset Creation

BraveATA was constructed from accepted 2025 papers at ICLR, ICML, and NeurIPS that contain theoretical analyses of LLMs. Candidate papers were screened for relevance, and selected papers were required to contain a central claim, an explicitly identifiable source statement, and proof-level reasoning.

The annotation pipeline combines expert annotation with LLM assistance. Final annotations were manually checked and revised by domain experts for semantic alignment, type correctness, and minimality. The dataset includes expert-validated natural-language theorem statements, informal proof sketches, Lean 4 source headers, and Lean 4 theorem statements.

## License

The BraveATA annotations, metadata, and benchmark structure are released under **CC0 1.0 Universal**.

Any source-paper excerpts or quoted material remain subject to the licenses and copyrights of the original papers. Users should cite the original source papers where applicable.
