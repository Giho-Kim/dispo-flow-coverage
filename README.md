# DiSPO flow matching and coverage-weighted training

Conditional outcome and action generators trained with flow matching, using the DiSPO Bellman target, conditional U-Net architecture, outcome scaling, EMA inference, and elite outcome selection. Coverage reweights training losses by the state and Bellman outcome. Both methods collect with random outcome draws. The implementation supports discrete and bounded continuous actions, accumulating uniform transition replay, full checkpoint resume, and fresh outcome/action generation during evaluation.

See [implementation and commands](shared_readout/dispo_fm/README.md).

## GPU setup

```bash
conda env create -f environment.yml
conda activate dispo-fm-gpu
python shared_readout/run.py train --device cuda --output shared_readout/fm_v3_results/coverage --method coverage --rounds 100
```

The Conda environment pins Python 3.11 and PyTorch 2.14.1 with CUDA 12.6 wheels. Training and evaluation default to `--device auto`, which uses CUDA when available and otherwise runs on CPU. Use `--device cuda` to require a GPU or `--device cuda:0` to select a specific GPU.

Garnet training runs evaluation every five rounds and at the final round. The trainer prints the results and saves them under `evaluation/roundNNN/` inside the run directory. Set `--eval-every N` to change the interval or `--eval-every 0` to disable automatic evaluation.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest shared_readout.dispo_fm.test_core -v
```

## Reference

Chuning Zhu et al., *Distributional Successor Features Enable Zero-Shot Policy Optimization*, NeurIPS 2024. Official repository: https://github.com/WEIRDLabUW/dispo

The `dispo_official` directory contains retrieved reference source from commit `03abef45836b1c238ff90b5b8e93132d3dc41e91`. The PyTorch flow-matching implementation is in `shared_readout/dispo_fm`. Its architecture, training choices, and FM-specific adaptations are documented there.
