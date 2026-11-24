# Third-party baselines

Not vendored in this mirror. Clone here before RelDDPM / TabDiff harness runs:

```bash
git clone --depth 1 https://github.com/ruc-datalab/RelDDPM.git
git clone --depth 1 https://github.com/MinkaiXu/TabDiff.git
pip install -r RelDDPM/requirements.txt
```

Paper-faithful TabDiff training still uses the official conda env in that clone (`tabdiff.yaml`). The lightweight adapter in `../baselines/tabdiff_adapter.py` is enough for the smoke harness.
