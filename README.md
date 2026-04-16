# LLMSynth

Research implementation of **LLMSynth**: LLM-augmented semantic constraint discovery paired with a diffusion-based tabular synthesizer (TabDDPM-style backbone). This repository contains **source code only**—no precomputed results, checkpoints, or bundled datasets.

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

If you use this code, please cite the associated paper (Neurocomputing / preprint as applicable).

## License

Research code provided as-is for reproducibility; see repository license if present.
