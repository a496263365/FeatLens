# FeatLens FSE Replication Package

This directory contains the FeatLens source code, filtered benchmark metadata, experiment runners, evaluation scripts, and the experimental outputs used in the paper. It is intended to support result verification and experiment reproduction, rather than to serve as a fully offline distribution containing all external benchmark sources, dependency data, and model weights.

## 1. Method Overview

FeatLens addresses dependency retrieval and context construction for repository-level code generation. Its central idea is to use repository features as an interface between task descriptions and code entities, and to derive a compact, reusable dependency context through graph reasoning.

The method consists of three stages:

1. **Feature-oriented repository indexing.** FeatLens combines static structural analysis with semantic representations, performs hierarchical function clustering, and summarizes each feature cluster into a natural-language description together with its associated function set.
2. **Task-conditioned seed graph construction.** Given a task description and target function signature, FeatLens matches relevant features, reranks the feature assigned to the target location, collects candidate functions, and constructs a task-level seed graph from fine-grained code dependencies.
3. **Semantic-structure fused graph calibration.** FeatLens expands the seed graph by one structural hop, ranks and filters nodes using semantic similarity, structural proximity, and graph propagation scores, and retains a compact Top-K dependency graph for code generation.

On DevEval and EvoCodeBench, FeatLens improves dependency retrieval coverage over sparse, dense, and graph-based retrieval baselines. It also reduces repository-level graph construction and online retrieval overhead while providing dependency contexts with high reuse value for downstream generation. Detailed quantitative results are reported in the paper and in the per-task records under `results/`.

## 2. Directory Structure

```text
FeatLens_FSE_Replication/
├── README.md
├── requirements.txt
├── .env.example
├── featlens_paths.py
├── src/
├── scripts/
├── data/
└── results/
```

### `src/`

The main FeatLens source code.

- `src/summarization/`: repository structure analysis, function representation, file and function clustering, feature description generation, and feature index construction.
- `src/search/`: code retrieval based on features, BM25, UniXcoder, and their combinations.
- `src/graph/`: seed graph construction, one-hop expansion, node ranking, dependency graph calibration, and graph analysis.
- `src/generation/`: prompt construction from retrieved dependency context and code generation through large language models.
- `src/statistics/`: code length, cyclomatic complexity, project scale, and result aggregation utilities.
- `src/utils/`: shared utilities for project path resolution and experiment input organization.

### `scripts/`

Experiment runners, result recomputation, and metric computation scripts.

- `scripts/retrieval/`: retrieval entry points, dependency graph scoring, DR/Precision/F1 computation, and result recomputation.
- `scripts/generation/`: post-processing, testing, and aggregation scripts for LocAgent, CodexGraph, Oracle, and related generation experiments.
- `scripts/metrics/`: DevEval Pass@1, DIR, dependency recall, and RepoScope matching implementations.
- `scripts/ablation/`: runners and aggregators for feature guidance, BM25, UniXcoder, and structural heuristic ablations.
- `scripts/repograph/`: RepoGraph experiment runners and generation evaluation scripts.
- `scripts/statistics/`: archived utilities for code length, complexity, and project statistics.

The scripts in `scripts/` are experiment orchestration and scoring tools. Some depend on implementation modules in `src/`, and others are retained as runners for specific experiment batches. Inspect each script's `--help` output before execution.

### `data/`

Filtered benchmark metadata used in the paper.

- `data/deveval/`: filtered DevEval project lists, complete task metadata, tasks with dependencies, and tasks without dependencies.
- `data/evocodebench/`: filtered EvoCodeBench project lists, task metadata, and tasks with dependencies.
- `data/**/projects/`: project-level task metadata, consistent with the aggregate JSONL files.
- The related Excel files contain project-scale statistics required by the paper.

`data/` does not include benchmark project source code or the complete dependency packages required for test execution. Source paths are configured through environment variables.

### `results/`

Experimental outputs and scoring records used in the paper.

- `results/retrieval/`: retrieval outputs, candidate graphs, per-task scores, and aggregate results for FeatLens, BM25, UniXcoder, RepoGraph, RepoScope, LocAgent, and CodexGraph.
- `results/generation/deveval/`: generated code, per-task test records, dependency invocation analyses, and metric summaries for DevEval methods.
- `results/ablation/deveval/`: ablation results for feature guidance, BM25, UniXcoder, and related graph processing conditions.
- `results/cost/deveval/`: context length, token statistics, and cost records.

`results/` contains compact experimental records. The full `results/` directory, including graph files, ENRE reports, and retained generation and evaluation artifacts, is available in [FeatLens_FSE_Replication.tar.zst](FeatLens_FSE_Replication.tar.zst) in this repository. The archive contains only `results/`; source code and other package files are provided separately in the repository. Full API request/response traces and temporary evaluator workspaces are intentionally omitted. New experiments should use separate output directories to avoid overwriting the archived records.

Download the archive using the file page's download button, or obtain it by cloning this repository. With GNU tar and Zstandard installed, extract it from the repository root:

```bash
tar --zstd -xf FeatLens_FSE_Replication.tar.zst
```

The archive extracts directly into `results/`, replacing any existing files with the same paths. Run the reproduction steps below from the repository root.

### Root Files

- `requirements.txt`: frozen Python package list for the FeatLens experiment environment.
- `.env.example`: template for APIs, external source trees, dependency data, the evaluation interpreter, and local model paths.
- `featlens_paths.py`: shared path resolution for package-internal paths, external paths, model paths, and temporary directories.

## 3. Reproduction Steps

### 3.1 Prepare the Python Environment

Python 3.10 is recommended. Create an isolated virtual environment:

```bash
cd /path/to/FeatLens_FSE_Replication
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the version matching the local CUDA environment before installing the remaining dependencies.

### 3.2 Configure Environment Variables

Copy the template and update it for the local environment:

```bash
cp .env.example .env
```

The following variables are required:

- `OPENAI_BASE_URL` and `OPENAI_API_KEY`: OpenAI-compatible model service endpoint and credential.
- `FEATLENS_DEVEVAL_SOURCE_ROOT`: root directory of the DevEval project sources.
- `FEATLENS_EVOCODEBENCH_SOURCE_ROOT`: root directory of the EvoCodeBench project sources.
- `FEATLENS_DEPENDENCY_DATA_ROOT`: root directory of the data required for DevEval dependency tests.
- `FEATLENS_PROJECT_LIST_ROOT`: directory containing experiment project lists or supplementary project spreadsheets.
- `FEATLENS_MINILM_MODEL`, `FEATLENS_BGE_MODEL`, and `FEATLENS_UNIXCODER_MODEL`: local model directories or Hugging Face model identifiers.

To load the variables into the current shell:

```bash
set -a
source .env
set +a
```

Model weights, benchmark project sources, and dependency test data are not distributed with this package and must be prepared separately.

### 3.3 Build the Feature Index

Generate repository structure analyses, function clusters, feature descriptions, `features.csv`, `methods.csv`, and ENRE reports. The batch entry point for DevEval is:

```bash
python src/summarization/run_deveval_batch.py --help
```

A single-project example entry point is:

```bash
python src/summarization/main_mopidy.py
```

Confirm the project source and output paths in `.env` before execution. Feature description generation requires an OpenAI-compatible endpoint.

### 3.4 Run Feature Retrieval

Use the feature index to retrieve candidate functions for each task and produce diagnostic records:

```bash
python src/search/batch_feature_based_search.py --help
```

The output should include task search results, feature matching diagnostics, and retrieval metrics. Additional baseline retrieval scripts are available under `src/search/`.

### 3.5 Construct and Calibrate Dependency Graphs

First construct the seed graphs:

```bash
python src/graph/batch_build_graph.py --help
```

Then perform one-hop expansion, semantic-structural scoring, Top-K ranking, and graph calibration:

```bash
python src/graph/batch_expand_and_rank_graph.py --help
```

The outputs should include `_ori.gml`, `_mid.gml`, and Top-K ranked `_rank.gml` files.

### 3.6 Evaluate Dependency Retrieval

Recompute DR, Precision, and F1 with the unified entity-matching logic:

```bash
python scripts/retrieval/graph/batch_compare_graph_recall.py --help
```

The scoring scripts for FeatLens feature retrieval and final graphs are:

```bash
python scripts/retrieval/featlens/recompute_seed.py
python scripts/retrieval/featlens/rescore.py
```

These scripts read the retained graph files and per-task metadata under `results/retrieval/` and write scoring outputs to the corresponding aggregate directories.

### 3.7 Run Code Generation

Convert calibrated dependency graphs into generation contexts and call the configured model:

```bash
python src/generation/dev_eval/batch_run_graph_rag.py --help
```

Generation outputs should be written to new directories and follow the same data structure as the records under `results/generation/deveval/featlens/`.

### 3.8 Compute Generation Metrics

Compute Pass@1 with the original DevEval test scripts:

```bash
python scripts/metrics/deveval/pass_k.py --help
```

Compute DIR with the dependency analysis script:

```bash
python scripts/metrics/deveval/parser/recall_k.py --help
```

Compute code length and cyclomatic complexity with:

```bash
python src/statistics/code_len_stats.py --help
python src/statistics/cyclomatic_complexity_stats.py --help
```

Final results should align with the per-task outputs, test logs, and aggregate files organized by method and model under `results/generation/deveval/`.

### 3.9 Reproduction Notes

- The existing `results/` directory is the original archival evidence for the paper and should not be overwritten.
- Path resolution is relative to the package root. External source trees, dependency data, and model paths are controlled through `.env`.
- Some baseline implementations may require additional external dependencies. This package preserves the runners and scoring logic used in the paper but does not include every third-party baseline implementation.
- To verify metrics only, the retained graphs, generation outputs, and per-task logs under `results/` can be rescored without invoking models.
- For a full reproduction, follow the sequence: feature index construction → feature retrieval → graph expansion and calibration → dependency retrieval evaluation → code generation → Pass@1/DIR computation.
