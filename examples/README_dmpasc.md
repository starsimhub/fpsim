# DMPA-SC Intervention Example

This example demonstrates the implementation of DMPA-SC (depot medroxyprogesterone acetate subcutaneous) contraceptive interventions in FPsim, as specified in [GitHub Issue #416](https://github.com/fpsim/fpsim/issues/416).

## Overview

The `example_dmpasc.py` script implements four scenarios:

1. **Baseline**: 2% annual increase in **injectable** users
2. **Scenario 1**: 3-month DMPA-SC scale-up from 1.5% (2025) to 5% (2040) in women under 20
3. **Scenario 2**: 6-month DMPA-SC scale-up from 1% (2030) to 5% (2040) in women under 20, with switching from 3-month DMPA-SC (26%) and traditional methods (20%)
4. **Scenario 3**: Combined scenario (identical to Scenario 2, includes both 3-month and 6-month)

## Enhanced Interventions

This example utilizes enhanced versions of existing interventions and the built-in switching intervention:

### Enhanced `change_initiation`

The existing `change_initiation` intervention has been enhanced with new optional parameters for age-restricted, targeted initiation with time-varying rates.

**New parameters:**
- `age_range` - Restrict to specific age range (e.g., (0, 20) for women under 20)
- `perc_of_eligible` - If True, `perc` applies to eligible women, not current users
- `target_method` - Direct new users to a specific method
- `final_perc` - Enable linear scale-up from `perc` to `final_perc`

**Example usage:**
```python
intv = fp.change_initiation(
    years=[2025, 2040],           # Intervention period
    age_range=(0, 20),            # Women under 20
    perc=0.015,                   # 1.5% initial rate
    final_perc=0.05,              # 5% final rate (linear scale-up)
    perc_of_eligible=True,        # Apply to eligible women
    target_method='dmpasc3',      # Target method
    annual=True,                  # Annual rates
)
```

### `method_switching`

Causes women to switch from specific contraceptive methods to a target method by modifying switching probabilities.

**Key features:**
- Multiple source methods
- Method-specific switching rates
- Age-group and postpartum-state aware
- One-time activation at specified year

**Example usage:**
```python
intv = fp.method_switching(
    year=2030,                                           # When to apply
    from_methods=['dmpasc3', 'wdraw', 'othtrad'],         # Source methods
    to_method='dmpasc6',                                 # Target method
    switch_prob={'dmpasc3': 0.26, 'wdraw': 0.20, 'othtrad': 0.20},
    annual=False,
)
```

## DMPA-SC Method Definition

DMPA-SC methods are typically introduced via `add_method()` by copying from an existing method
(usually `inj`) and overriding a small set of parameters:

```python
fp.add_method(
    year=2025,
    method=None,  # copy_from provides the base method
    copy_from='inj',
    split_shares=0.0,
    method_pars={
        'name': 'dmpasc3',
        'label': 'DMPA-SC 3-month',
        'csv_name': 'DMPA-SC 3-month',
        'dur_use': ss.lognorm_ex(mean=2, std=1),  # placeholder base duration
        'rel_dur_use': 2.0,                       # placeholder: 2× longer duration vs injectables
    },
)
```

**Key parameter: `rel_dur_use`**
- Multiplicative scaling factor for duration of use
- `rel_dur_use=2.0` means women stay on the method twice as long
- Applied after the base duration distribution is sampled

## Running the Example

```bash
python examples/example_dmpasc.py
```

This will:
1. Run all five scenarios
2. Generate a single dashboard figure: `dmpasc_dashboard.png`
3. Print summary statistics

### Plot display (optional)

By default the script runs non-interactively (it saves the dashboard and exits). To show the figure:

```bash
python examples/example_dmpasc.py --show
```

To skip saving (useful for quick runs):

```bash
python examples/example_dmpasc.py --no-save
```

## Expected Results

### Baseline Scenario
- Steady 2% annual growth in **injectable** users (new initiators are directed to `inj`)
- No DMPA-SC methods available
- Establishes reference trajectory for comparison

### Scenario 1 (3-month DMPA-SC)
- New method introduced in 2025
- Gradual uptake in women under 20, reaching 5% by 2040
- Modest increase in mCPR due to improved retention (2x duration)
- Small reduction in births compared to baseline

### Scenario 2 (6-month DMPA-SC)
- Both 3-month (2025) and 6-month (2030) methods available
- Switching dynamics reduce 3-month usage after 2030
- 6-month reaches 5% in target age group by 2040
- Greater mCPR increase due to even longer retention
- More births averted than Scenario 1

### Scenario 3 (Combined)
- Identical to Scenario 2 (demonstrates same interventions)
- Useful for comparing different implementation strategies

## Key Implementation Details

### Baseline Growth Mechanism
Uses `change_initiation` targeted to injectables:
```python
fp.change_initiation(
    years=[start, stop],
    perc=0.02,
    annual=True,
    perc_of_eligible=True,
    target_method='inj',
)
```
- `perc=0.02`: 2% growth rate
- `annual=True`: converted to a per-timestep rate internally (based on `dt`)
- `perc_of_eligible=True`: applies the fraction to eligible non-users (rather than scaling from current users)
- `target_method='inj'`: directs new users to injectables

### Method Introduction
Uses existing `add_method` intervention:
```python
fp.add_method(
    year=2025,
    method=None,
    copy_from='inj',
    split_shares=0.0,
    method_pars={'name': 'dmpasc3', 'rel_dur_use': 2.0},
)
```
- `copy_from='inj'`: Copies switching behavior from injectables
- `split_shares=0.0`: Don't automatically split existing shares (use targeted interventions instead)

### Duration Scaling
The `rel_dur_use` parameter scales the sampled duration:
```python
duration_on_method = base_duration × rel_dur_use
```
- Base duration from injectable method's distribution
- `rel_dur_use=2.0` doubles the time women stay on method
- Improves contraceptive continuation and effectiveness

## Customization

### Modify Scale-Up Trajectory
Change initiation rates or years:
```python
fp.change_initiation(
    years=[2020, 2035],      # Different time period
    age_range=(0, 20),       # Women under 20
    perc=0.01,               # Start at 1%
    final_perc=0.10,         # End at 10% (linear scale-up)
    perc_of_eligible=True,   # Apply to eligible women
    target_method='dmpasc3', # Target method
    annual=True,
)
```

### Change Age Targeting
Modify age range:
```python
age_range=(15, 25)  # Target ages 15-25
age_range=(0, 100)  # Target all ages
```

### Adjust Switching Rates
Modify switching probabilities:
```python
switch_prob={'inj': 0.20, 'wdraw': 0.10}  # Different rates
```

### Change Duration Scaling
Modify relative duration:
```python
rel_dur_use=1.5   # 50% longer
rel_dur_use=3.0   # 3x longer
```

## Validation

Run the test suite to validate implementation:
```bash
pytest tests/test_interventions.py -v
```

Key tests:
- `test_age_restricted_initiation_basic()` - Age filtering
- `test_age_restricted_initiation_time_varying()` - Scale-up trajectory
- `test_method_switching_basic()` - Switching functionality
- `test_method_switching_multiple_sources()` - Multiple source methods
- `test_dmpasc_scenario_integration()` - Full scenario integration

## Performance Considerations

- Larger `n_agents` (10,000+) provides smoother trajectories
- Longer simulation periods (40+ years) better demonstrate scale-up
- Multiple scenarios can be run in parallel if needed

## References

- GitHub Issue: https://github.com/fpsim/fpsim/issues/416
- FPsim Documentation: https://docs.fpsim.org
- Related interventions: `change_initiation`, `add_method`, `method_switching`, `update_methods`

## Authors

Implementation based on specifications from:
- @mzimmermann-IDM (scenario design)
- @pausz (technical clarifications)
- FPsim development team
