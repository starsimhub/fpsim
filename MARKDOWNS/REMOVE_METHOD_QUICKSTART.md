# Remove Method - Quick Start Guide

## What is it?

The `remove_method` functionality allows you to remove contraceptive methods from your FPsim simulation. This is useful for modeling scenarios where methods become unavailable due to supply disruptions, policy changes, or product discontinuation.

**New:** Comprehensive visualization tools to analyze and communicate the impact of method removals.

## Basic Usage

### Option 1: Using MethodIntervention (Recommended)

```python
import fpsim as fp

# Setup parameters
pars = dict(n_agents=5000, start_year=2000, end_year=2030)

# Create intervention to remove implants in 2015
mod = fp.MethodIntervention(year=2015, label='Remove Implants')
mod.remove_method('impl', reassign_to='inj')

# Build and run
intv = mod.build()
sim = fp.Sim(pars=pars, location='senegal', interventions=intv)
sim.run()
```

### Option 2: Direct Removal

```python
import fpsim as fp

# Initialize simulation
sim = fp.Sim(pars=pars, location='senegal')
sim.init()

# Remove method directly
sim.connectors.contraception.remove_method('IUDs', reassign_to='Injectables')

# Run simulation
sim.run()
```

### Option 3: Using update_methods Intervention

```python
import fpsim as fp

# Create intervention
intv = fp.update_methods(
    year=2020,
    remove_method={
        'method_label': 'Other modern',
        'reassign_to': 'Pill'
    },
    verbose=True
)

sim = fp.Sim(pars=pars, location='senegal', interventions=intv)
sim.run()
```

## Method Names Reference

Use these short names for the `method` parameter:

| Short Name | Full Name |
|------------|-----------|
| `'pill'` | Pills |
| `'iud'` | IUDs |
| `'inj'` | Injectables |
| `'impl'` | Implants |
| `'cond'` | Condoms |
| `'btl'` | Female sterilization (tubal ligation) |
| `'wdraw'` | Withdrawal |
| `'othtrad'` | Other traditional |
| `'othmod'` | Other modern |

For the `reassign_to` parameter, you can also use the full labels like:
- `'Pill'`, `'IUDs'`, `'Injectables'`, `'Implants'`, etc.
- `'none'` - to stop contraception (users become non-users)

## Visualizing Method Removal Impact

### Quick Visualization

Generate a comprehensive summary figure with one function call:

```python
from plots import create_removal_summary_figure

# Run baseline and removal simulations
baseline_sim = fp.Sim(pars=pars, location='senegal')
baseline_sim.run()

mod = fp.MethodIntervention(year=2015, label='Remove Withdrawal')
mod.remove_method('wdraw')
intv = mod.build()
removal_sim = fp.Sim(pars=pars, location='senegal', interventions=intv)
removal_sim.run()

# Generate comprehensive summary figure
create_removal_summary_figure(
    baseline_sim, removal_sim,
    start_year=2000, end_year=2030,
    removal_year=2015, location='senegal',
    save_path='figures/removal_summary.png'
)
```

This creates a 5-panel figure showing:
1. Removed method usage timeline
2. CPR comparison
3. mCPR comparison
4. Where users redistributed
5. Final method mix

### Individual Plots

For more control, generate specific plots:

```python
from plots import (plot_method_removal_impact, 
                   plot_method_redistribution,
                   plot_removed_method_timeline)

# Impact on removed method and CPR
plot_method_removal_impact(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal',
                          save_path='figures/impact.png')

# Where did users go?
plot_method_redistribution(baseline_sim, removal_sim, 2015, 'senegal',
                          save_path='figures/redistribution.png')

# Timeline showing before/after
plot_removed_method_timeline(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal',
                            save_path='figures/timeline.png')
```

### Complete Example with Visualizations

```python
import fpsim as fp
from plots import create_removal_summary_figure
from pathlib import Path

# Setup
pars = dict(n_agents=5000, start_year=2000, end_year=2030)
location = 'senegal'
removal_year = 2015

# Create figures directory
Path('figures').mkdir(exist_ok=True)

# Run baseline
print("Running baseline...")
baseline_sim = fp.Sim(pars=pars, location=location, label='Baseline')
baseline_sim.run()

# Run with removal
print("Running with method removal...")
mod = fp.MethodIntervention(year=removal_year, label='Remove Implants')
mod.remove_method('impl', reassign_to='inj')
intv = mod.build()
removal_sim = fp.Sim(pars=pars, location=location, interventions=intv, 
                     label='With Removal')
removal_sim.run()

# Generate visualizations
print("Generating visualizations...")
create_removal_summary_figure(
    baseline_sim, removal_sim,
    pars['start_year'], pars['end_year'],
    removal_year, location,
    save_path='figures/removal_summary.png'
)

print("✓ Complete! See figures/removal_summary.png")
```

## Common Use Cases

### Supply Chain Disruption
```python
# Model temporary unavailability of a method
mod = fp.MethodIntervention(year=2020, label='Supply Disruption')
mod.remove_method('impl', reassign_to='inj')
```

**Analysis Question:** Where do implant users go when implants are unavailable?
**Visualization:** Use `plot_method_redistribution()` to see the answer.

### Product Discontinuation
```python
# Model permanent removal of a method
mod = fp.MethodIntervention(year=2025, label='Product Discontinued')
mod.remove_method('othmod', reassign_to='pill')
```

**Analysis Question:** What's the impact on overall contraceptive prevalence?
**Visualization:** Use `plot_method_removal_impact()` to see CPR changes.

### Policy Restriction
```python
# Model policy change restricting a method
mod = fp.MethodIntervention(year=2018, label='Policy Restriction')
mod.remove_method('wdraw', reassign_to='cond')
```

**Analysis Question:** How does the method mix evolve over time?
**Visualization:** Use `plot_removed_method_timeline()` for timeline comparison.

### Multiple Changes
```python
# Remove method while improving alternatives
mod = fp.MethodIntervention(year=2020, label='Phase Out Program')
mod.remove_method('impl', reassign_to='inj')
mod.set_duration_months('inj', 36)  # Improve injectable continuation
mod.set_efficacy('inj', 0.99)       # Improve injectable efficacy
```

**Analysis Question:** What's the comprehensive impact of the program?
**Visualization:** Use `create_removal_summary_figure()` for full overview.

## Preview Your Intervention

Always check what your intervention will do before running:

```python
mod = fp.MethodIntervention(year=2025)
mod.remove_method('impl', reassign_to='inj')

# Preview the configuration
print(mod.preview())
```

Output:
```python
{
    'year': 2025,
    'remove_method': {
        'method_label': 'Implants',
        'reassign_to': 'Injectables'
    },
    'efficacy': None,
    'duration_months': None,
    ...
}
```

## Combining with Other Interventions

You can combine method removal with other changes:

```python
# Remove implants and improve injectable duration
mod = fp.MethodIntervention(year=2020, label='Comprehensive Program')
mod.remove_method('impl', reassign_to='inj')
mod.set_duration_months('inj', 36)  # Improve injectable continuation
mod.set_efficacy('inj', 0.99)       # Improve injectable quality

intv = mod.build()
sim = fp.Sim(pars=pars, interventions=intv)
sim.run()
```

## Important Notes

✅ **Do:**
- Always specify what method users should be reassigned to
- Use `reassign_to='none'` if users should stop contraception
- Preview your intervention before running
- Generate visualizations to understand impact
- Compare with a baseline simulation

❌ **Don't:**
- Try to remove the 'none' method (it's the non-use state)
- Assume removed methods can be automatically re-added
- Remove methods that don't exist (it will raise an error)
- Forget to create a baseline for comparison when using visualization tools

## Checking Results

### Check Methods Programmatically

```python
# Check which methods remain
cm = sim.connectors.contraception
print(f"Remaining methods: {list(cm.methods.keys())}")
print(f"Total methods: {cm.n_methods}")

# Check method mix
for method_name, method in cm.methods.items():
    if method_name == 'none':
        continue
    n_users = (sim.people.fp.method == method.idx).sum()
    pct = 100 * n_users / len(sim.people)
    print(f"{method.label}: {n_users} users ({pct:.2f}%)")
```

### Check Results Visually

The plotting functions automatically generate professional figures showing:

- **Removal Impact Plot**: Timeline of removed method usage + CPR comparison
- **Redistribution Plot**: Bar charts showing where users went
- **Timeline Plot**: Stacked area charts comparing before/after
- **Summary Figure**: Comprehensive 5-panel overview

All plots saved at 300 DPI for publication quality.

## Running the Examples

### Option 1: Standalone Example
```bash
cd examples
python example_remove_method.py
```

Runs 3 scenarios and generates all visualizations automatically.

### Option 2: Integration Example
```bash
cd examples
python example_method_intervention_usage.py
```

With `usecases = [0, 5]` and `plot_compare = [[0,5]]`, this automatically generates specialized removal plots.

## Understanding the Visualizations

### 1. Removal Impact Plot (`plot_method_removal_impact`)

**Left Panel: Removed Method Timeline**
- Shows usage of the removed method dropping to zero
- Vertical line marks removal year
- Annotation box highlights removal point

**Right Panel: CPR Comparison**
- Compares overall contraceptive prevalence
- Shows if removal affected overall contraception use
- Displays final CPR change

**Use this when:** You want to see the immediate impact of removal on both the specific method and overall contraception.

### 2. Redistribution Plot (`plot_method_redistribution`)

**Left Panel: Change Analysis**
- Horizontal bars show % change in each method
- Green = methods that gained users
- Red = methods that lost users
- Sorted by magnitude of change

**Right Panel: Method Mix Comparison**
- Stacked bars comparing baseline vs. after removal
- Shows complete method mix picture
- Easy to see redistribution patterns

**Use this when:** You want to understand where users of the removed method went.

### 3. Timeline Plot (`plot_removed_method_timeline`)

**Top Panel: Baseline**
- All methods including the one to be removed
- Removed method highlighted in red
- Vertical line marks removal year

**Bottom Panel: After Removal**
- Remaining methods only
- Same time scale for easy comparison
- Shows how method mix evolves

**Use this when:** You want a clear before/after visual comparison over time.

### 4. Summary Figure (`create_removal_summary_figure`)

**5-Panel Comprehensive Overview:**
1. Removed method timeline with annotations
2. CPR over time comparison
3. mCPR over time comparison
4. Method redistribution bar chart
5. Final method mix comparison bars

**Use this when:** You need a complete overview for presentations or publications.

## Troubleshooting

### Plot Generation Issues

**Problem:** Plots show no removed methods
- **Solution:** Make sure you're comparing a baseline simulation (all methods) with an intervention simulation (method removed)

**Problem:** Method names not displaying correctly
- **Solution:** The functions auto-detect removed methods by comparing sims. Ensure both sims use the same location/setup.

**Problem:** Figures directory not created
- **Solution:** The plotting functions create the directory automatically, but you can create it manually: `Path('figures').mkdir(exist_ok=True)`

### Method Removal Issues

**Problem:** Method not found error
- **Solution:** Check method name spelling. Use short names like `'impl'` or full labels like `'Implants'`

**Problem:** Reassignment method not found
- **Solution:** Make sure the target method exists and is spelled correctly

**Problem:** Simulation crashes after removal
- **Solution:** Make sure simulation is initialized before direct removal

## Getting Help

### Documentation
- `examples/PLOTTING_GUIDE.md` - Complete plotting documentation
- `MARKDOWNS/REMOVE_METHOD_FEATURE.md` - Full feature documentation
- `examples/example_remove_method.py` - Working code examples
- `examples/example_method_intervention_usage.py` - Integration examples

### Quick Checks
- Verify method names are spelled correctly
- Ensure reassignment method exists
- Check simulation is initialized before direct removal
- Confirm both baseline and intervention sims for visualization

### Example Files Location
All examples are in the `examples/` directory:
- `example_remove_method.py` - Method removal scenarios
- `example_method_intervention_usage.py` - Multiple intervention types
- `plots.py` - All plotting functions
- `PLOTTING_GUIDE.md` - Visualization documentation

## Next Steps

1. **Try the basic example:** Run `example_remove_method.py`
2. **Experiment with different methods:** Change which method is removed
3. **Generate visualizations:** Add plotting calls to your code
4. **Compare scenarios:** Run multiple removal scenarios and compare
5. **Read the full docs:** Check `REMOVE_METHOD_FEATURE.md` for details

Happy modeling! 🚀
