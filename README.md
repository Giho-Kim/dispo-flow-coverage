# DiSPO flow matching and coverage collection

Conditional outcome and action generators trained with flow matching, using the DiSPO Bellman target, conditional U-Net architecture, outcome scaling, EMA inference, and elite outcome selection. Coverage chooses additional data collection targets. The implementation supports discrete and bounded continuous actions, accumulating uniform transition replay, full checkpoint resume, and fresh outcome/action generation during evaluation.

See [implementation and commands](shared_readout/dispo_fm/README.md).

## Validation

```bash
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest shared_readout.dispo_fm.test_core -v
```

Six tests and small-network Garnet/continuous training, resume, and evaluation passed. The full default U-Net with batch 2048 exceeded this workspace's memory on the first attempt. Generation was subsequently changed to skip gradient recording; the retry was interrupted before completion. A 100-round v2 result is not available.

## Reference

Chuning Zhu et al., *Distributional Successor Features Enable Zero-Shot Policy Optimization*, NeurIPS 2024. Official repository: https://github.com/WEIRDLabUW/dispo

The `dispo_official` directory contains retrieved reference source from commit `03abef45836b1c238ff90b5b8e93132d3dc41e91`. The new NumPy implementation is in `shared_readout/dispo_fm`. Its architecture, training choices, and FM-specific adaptations are documented there.
