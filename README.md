# LLMSynth

Research implementation of **LLMSynth**: LLM-augmented semantic constraint discovery paired with a diffusion-based tabular synthesizer (TabDDPM-style backbone).

## What is included

| Module | Role |
|--------|------|
| `llm_discovery.py` | Schema + sample → candidate constraints (Anthropic API); optional disk cache under `cache/` (gitignored) |
| `constraints.py` | Constraint types, checking, violation rates |
| `synthesizer.py` | Training and sampling with constraint guidance |
| `datasets.py` | Loaders that **download** public benchmarks via scikit-learn / OpenML (no data files in repo) |
| `metrics.py` | Evaluation metrics |
| `run_experiments.py` | End-to-end benchmark driver |
| `benchmark_config.py` | Dataset / method configuration |

## Requirements

- Python 3.10+
- CUDA optional (CPU training is slow but supported for smoke tests)

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`.
2. Set `ANTHROPIC_API_KEY` for live LLM calls. For a **dry run without API access**, use `--use-mock` (see below).
3. Optional: set `LLMSYNTH_ANTHROPIC_MODEL` to the exact Anthropic model id your key should use (defaults to the Claude 3.5 Sonnet snapshot named in the manuscript).

Do **not** commit `.env`.

## Datasets

Benchmarks are fetched automatically (e.g., OpenML, UCI via `sklearn.datasets`). No dataset files are stored in this repository.

## Running

Full benchmark (requires GPU time + API budget):

```bash
python run_experiments.py
```

Single dataset, mock discovery (no API):

```bash
python run_experiments.py --dataset adult --use-mock --seeds 1 --epochs 2
```

Reuse cached LLM responses (after a successful run populated `cache/` locally):

```bash
python run_experiments.py --use-cache
```

Raw tables and LaTeX snippets are written under `--output-dir` (default: `results_paper3/`), which is gitignored.

## Citation

If this implementation supports your work, cite the **peer-reviewed article** using the publisher’s official bibliographic record (authors, title, venue, volume/issue, pages, DOI) once that reference exists. **Do not** treat this README as the bibliographic record, and **do not** infer venue or citation metadata from this repository.

## Manuscript ↔ code parity (checklist)

Keep this list in sync when the paper’s experiments or tables change.

| Area | Current repo state | Typical follow-up |
|------|-------------------|-------------------|
| **Benchmark datasets** | Defaults cover five tasks (`adult`, `credit`, `heart`, `diabetes`, `wine`) via `benchmark_config.LLMSYNTH_DATASETS`, `datasets._CANONICAL_DATASETS`, and `run_experiments.py` CLI defaults. | If the manuscript adds benchmarks (e.g. banking / credit-default / regression housing splits), add loaders plus ground-truth constraint specs in `datasets.py`, then extend the lists above. |
| **Synthesis baselines** | Runner registers `LLMSynth`, `TabDDPM`, `ManualConstraints`, `PostHocRepair` (`run_experiments.ALL_METHODS`). | Any extra generators named in the paper (e.g. score-based or LLM tabular baselines) need adapters in `synthesizer.py` and wiring in `_create_synthesizer`. |
| **Discovery baselines** | LLM discovery + optional mock/cache; no TANE/Hydra drivers in-tree. | If the discovery table must be reproduced end-to-end, add baseline pipelines or document external tooling + how outputs feed `metrics.constraint_discovery_quality`. |
| **Validation threshold** | `LLMSYNTH_VALIDATION_THRESHOLD` / `violation_threshold=0.05` should match the paper’s single-threshold statistical gate (tau). | After any TeX change to tau or validation logic, update `benchmark_config.py`, `llm_discovery.py`, and call sites in `run_experiments.py`. |

## License

Research code provided as-is for reproducibility; see repository license if present.
