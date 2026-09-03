# Calibration workflow

This document describes the process for recalibrating FPsim when model logic changes.

## When to recalibrate

Recalibration is needed when:
- The pregnancy/conception pipeline changes (e.g., switching from single-check miscarriage to per-step p_loss)
- Sexual activity or contraception update logic changes
- New data sources are integrated
- A new location is added

Recalibration is NOT needed for:
- Bug fixes that restore intended behavior
- Adding new result tracking that doesn't affect dynamics
- Documentation or test changes

## Recalibration steps

### 1. Save current baselines

Before making changes, save the current calibration parameters as a baseline:

```bash
python fpsim/locations/compare_calib_pars.py --save
```

This saves a JSON snapshot per location in `fpsim/locations/calib_baselines/`.

### 2. Run calibration

For all locations (~45 min total):
```bash
SCIRIS_BACKEND=agg python fpsim/locations/calibrate_all.py
```

For a single location (~5 min):
```bash
SCIRIS_BACKEND=agg python fpsim/locations/calibrate_all.py --location kenya
```

Results are saved as `.obj` files in `fpsim/locations/calib_results/` as each location completes, so partial runs are not lost.

### 3. Compare parameters

After calibration, compare old vs new parameters:

```bash
python fpsim/locations/compare_calib_pars.py --verbose
```

This flags parameters that changed by more than 30% (configurable with `--tolerance`).

**Expected changes:**
- `exposure_age` and `exposure_parity` curves: these have many free parameters that trade off, so individual values can shift substantially (>50%) while the overall curve shape stays similar. Focus on the overall shape, not individual points.
- `exposure_factor`, `fecundity_low/high`: these are the main scalars. Changes >30% suggest a significant model behavior change.
- `prob_use_intercept/trend_par`: these control contraception uptake trends. Large changes suggest the contraception pipeline was affected.
- `spacing_pref`: controls birth spacing behavior. Changes here interact strongly with sexual activity logic.

**Red flags:**
- Scalar parameters changing by >2x
- `exposure_factor` or `fecundity` moving to bound limits
- Best mismatch significantly worse than previous calibration

### 4. Visual inspection

Run the calibrated model and inspect plots:

```bash
python fpsim/locations/run_calibrated_location.py kenya
```

This generates validation plots comparing model output to data for:
- mCPR trends
- Age-specific fertility rates (ASFR)
- Total fertility rate (TFR)
- Birth spacing distribution
- Age at first birth
- Method mix

### 5. Update location files

Update all location `.py` files from saved calibration results:

```bash
python fpsim/locations/calibrate_all.py --update
```

Or a single location:
```bash
python fpsim/locations/calibrate_all.py --update --location kenya
```

This rewrites each location's `make_calib_pars()` function with the best parameters.

### 6. Save new baselines

```bash
python fpsim/locations/compare_calib_pars.py --save
```

### 7. Update parameter regression snapshot

```bash
python tests/test_parameters.py --save-par-snapshot kenya
```

## Key files

| File | Purpose |
|------|---------|
| `fpsim/locations/calibrate_all.py` | Run calibration and update location files |
| `fpsim/locations/compare_calib_pars.py` | Compare parameters between versions |
| `fpsim/locations/run_calibrated_location.py` | Run and plot a calibrated model |
| `fpsim/locations/calib_results/*.obj` | Saved calibration results (best pars + mismatch) |
| `fpsim/locations/calib_baselines/*.json` | Parameter baselines for regression checking |
| `fpsim/locations/<country>/<country>.py` | Location-specific calibrated parameters |
| `tests/test_parameters.py` | Parameter coverage and regression tests |
| `tests/par_snapshot_<country>.json` | Effective parameter snapshots |
