# LLMSynth

Research implementation of **LLMSynth**: LLM-augmented semantic constraint discovery paired with a diffusion-based tabular synthesizer (TabDDPM-style backbone).

## What is included

| Module | Role |
|--------|------|
| `llm_discovery.py` | Schema + sample → candidate constraints (Anthropic API); optional disk cache under `cache/` (gitignored) |
| `constraints.py` | Constraint types, checking, violation rates |
| `synthesizer.py` | Training and sampling with constraint guidance |
| `baselines/` | **RelDDPM** (controllable) and **TabDiff** (fidelity) adapters |
| `datasets.py` | Loaders that **download** public benchmarks via scikit-learn / OpenML (no data files in repo) |
| `metrics.py` | Evaluation metrics |
| `run_experiments.py` | End-to-end benchmark driver |
| `benchmark_config.py` | Dataset / method configuration |
| `third_party/` | Optional clones of [RelDDPM](https://github.com/ruc-datalab/RelDDPM) and [TabDiff](https://github.com/MinkaiXu/TabDiff) |

## Requirements

- Python 3.10+
- CUDA optional (CPU training is slow but supported for smoke tests)

```bash
pip install -r requirements.txt
```

### Third-party baselines

```bash
cd third_party
git clone --depth 1 https://github.com/ruc-datalab/RelDDPM.git
git clone --depth 1 https://github.com/MinkaiXu/TabDiff.git
pip install -r RelDDPM/requirements.txt
```

RelDDPM imports the official `ddpm` package from that clone. TabDiff harness runs use the lightweight adapter in `baselines/tabdiff_adapter.py`; paper-faithful TabDiff training still uses the official conda env (`third_party/TabDiff/tabdiff.yaml`).

## Configuration

1. Copy `.env.example` to `.env`.
2. Set `ANTHROPIC_API_KEY` for live LLM calls. For a **dry run without API access**, use `--use-mock` (see below).
3. Optional: set `LLMSYNTH_ANTHROPIC_MODEL` to the exact Anthropic model id your key should use (defaults to the Claude 3.5 Sonnet snapshot named in the manuscript).

Do **not** commit `.env`. RelDDPM / TabDiff-only runs do **not** require an Anthropic key.

## Datasets

Benchmarks are fetched automatically (e.g., OpenML, UCI via `sklearn.datasets`). No dataset files are stored in this repository.

Eight manuscript datasets: `adult`, `credit`, `heart`, `diabetes`, `wine`, `bank`, `default`, `california_housing`.

## Running

Full benchmark (requires GPU time + API budget):

```bash
python run_experiments.py
```

### New baselines: RelDDPM + TabDiff

Local smoke (500-row Adult subsample, 2 epochs):

```bash
python scripts/smoke_relddpm_tabdiff.py
```

Harness run (mock LLM discovery; real synthesis metrics):

```bash
python run_experiments.py --method RelDDPM TabDiff --dataset adult --seeds 1 --epochs 2 --use-mock
```

Full grid (8 datasets × 3 seeds, mock discovery):

```bash
python run_experiments.py --method RelDDPM TabDiff --dataset all --seeds 3 --epochs 200 --use-mock --output-dir results_relddpm_tabdiff
```

**RelDDPM** is controllable: `run_experiments.py` passes **ground-truth hard constraints** (same predicates as ManualConstraints), not LLM-discovered rules. A classifier controller is trained on satisfying vs perturbed rows, then classifier guidance steers sampling.

**TabDiff** is unconstrained fidelity SOTA-style baseline; non-zero hard violations are expected.

Single dataset, mock discovery (no API):

```bash
python run_experiments.py --dataset adult --use-mock --seeds 1 --epochs 2
```

Reuse cached LLM responses (after a successful run populated `cache/` locally):

```bash
python run_experiments.py --use-cache
```

Raw tables and LaTeX snippets are written under `--output-dir` (default: `results_paper3/`), which is gitignored.

## GCP helpers (optional)

Project `knu-fog-experiments`, zone `us-central1-a` (override with env vars). No secrets in scripts.

```bash
export GCP_INSTANCE=llmsynth-gpu   # your VM name
bash scripts/gcp/rsync_pack.sh
bash scripts/gcp/remote_run.sh    # smoke + full grid with tee
bash scripts/gcp/stop_vm_reminder.sh
```

Pull results:

```bash
gcloud compute scp --recurse --project knu-fog-experiments --zone us-central1-a \
  llmsynth-gpu:~/llmsynth/2_Code/results_relddpm_tabdiff ./results_relddpm_tabdiff
```

## What you can reproduce from this repository

The entry point is `run_experiments.py`. By default it runs the methods defined there on the benchmark datasets named in `benchmark_config.py` under `LLMSYNTH_DATASETS`. Use `--dataset`, `--method`, `--seeds`, and `--epochs` to match the runs described in the paper’s experimental section. Some comparisons may rely on external baselines or additional datasets described only in the PDF; this artifact contains the LLMSynth implementation and the wired benchmarks/methods above.

## Citation

If you use this code or the method, please cite:

```bibtex
@article{kang2026llmsynth,
  title   = {Making Synthetic Tables Semantically Valid via {LLM} Constraints},
  author  = {Kang, Nathaniel},
  journal = {Information Systems},
  year    = {2026},
  note    = {To appear}
}
```

Update volume, pages, and DOI when Elsevier posts them.

## License

Research code provided as-is for reproducibility; see repository license if present.
