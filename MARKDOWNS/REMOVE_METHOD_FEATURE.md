# Remove Method Feature Implementation

## Overview

Implemented a comprehensive `remove_method` functionality in FPsim that allows users to dynamically remove contraceptive methods from simulations. This mirrors the existing `add_new_method` functionality and is useful for modeling scenarios where methods become unavailable due to supply disruptions, product discontinuation, or policy changes.

**New in this update:** Comprehensive visualization tools for analyzing method removal impact.

## Implementation Details

### 1. Core Method in `methods.py`

Added `remove_method()` method to the `ContraceptiveChoice` class:

**Key Features:**
- Removes a method from the simulation by label or name
- Automatically reassigns users currently on that method to a specified alternative
- Re-indexes remaining methods (shifts indices down for methods after the removed one)
- Contracts the switching matrix to remove rows and columns for the removed method
- Updates method mix arrays and renormalizes probabilities
- Handles both people state and switching matrices correctly

**Parameters:**
- `method_label` (str): The label or name of the method to remove
- `reassign_to` (str): Method to reassign current users to (default: 'none')

**Example:**
```python
sim.connectors.contraception.remove_method('Implants', reassign_to='Injectables')
```

### 2. Helper Method `_contract_switching_matrix()`

Handles the complex logic of contracting nested switching matrix structures:
- Removes method from method_idx lists
- Shifts indices greater than removed_idx down by 1
- Removes probabilities for the removed method from all transitions
- Renormalizes all probability distributions
- Handles multiple switching matrix structures (event types, age bins, postpartum)

### 3. Intervention Support in `interventions.py`

Added `remove_method` parameter to the `update_methods` intervention:

**Parameters:**
- `remove_method` (dict): Configuration for removing a method
  - `method_label`: The label or name of the method to remove
  - `reassign_to`: (optional) The method to reassign current users to (default: 'none')

**Example:**
```python
intv = fp.update_methods(
    year=2015,
    remove_method={'method_label': 'Implants', 'reassign_to': 'Injectables'},
    verbose=True
)
```

### 4. User-Friendly Wrapper in `wrappers.py`

Added `remove_method()` method to the `MethodIntervention` class for easy-to-use API:

**Features:**
- Method chaining support
- Automatic method name validation and normalization
- Integration with preview() for inspection
- Clear documentation with real-world use cases

**Example:**
```python
mod = fp.MethodIntervention(year=2024, label='Implant Discontinuation')
mod.remove_method('impl', reassign_to='inj')
intv = mod.build()
sim = fp.Sim(pars=pars, interventions=intv)
sim.run()
```

### 5. Visualization Tools in `plots.py`

Added comprehensive plotting functions for analyzing method removal impact:

**Available Plotting Functions:**

1. **`plot_method_removal_impact()`** - Shows removed method usage timeline and CPR impact
2. **`plot_method_redistribution()`** - Analyzes where users of removed method redistributed
3. **`plot_removed_method_timeline()`** - Side-by-side stacked area charts showing before/after
4. **`create_removal_summary_figure()`** - Comprehensive 5-panel summary figure

**Example:**
```python
from plots import plot_method_removal_impact, create_removal_summary_figure

# Generate visualizations
plot_method_removal_impact(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal')
create_removal_summary_figure(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal')
```

See `examples/PLOTTING_GUIDE.md` for complete documentation of all visualization functions.

## Usage Examples

### Example 1: Remove Method via Intervention

```python
import fpsim as fp

# Create intervention to remove implants in 2015
mod = fp.MethodIntervention(year=2015, label='Remove Implants')
mod.remove_method('impl', reassign_to='inj')

# Build and run
intv = mod.build()
sim = fp.Sim(pars=pars, location='senegal', interventions=intv)
sim.run()
```

### Example 2: Direct Removal via Connector

```python
import fpsim as fp

# Initialize simulation
sim = fp.Sim(pars=pars, location='senegal')
sim.init()

# Remove IUDs directly
cm = sim.connectors.contraception
cm.remove_method('IUDs', reassign_to='Injectables')

# Run simulation
sim.run()
```

### Example 3: Preview Before Running

```python
mod = fp.MethodIntervention(year=2025, label='Method Removal')
mod.remove_method('othmod', reassign_to='pill')

# Preview the intervention
print(mod.preview())
# Output:
# {
#   'year': 2025,
#   'remove_method': {'method_label': 'Other modern', 'reassign_to': 'Pill'},
#   'label': 'Method Removal',
#   ...
# }
```

### Example 4: With Comprehensive Visualizations

```python
import fpsim as fp
from plots import (plot_method_removal_impact, plot_method_redistribution,
                   plot_removed_method_timeline, create_removal_summary_figure)

# Setup
pars = dict(n_agents=5000, start_year=2000, end_year=2030)

# Run baseline (all methods)
baseline_sim = fp.Sim(pars=pars, location='senegal')
baseline_sim.run()

# Run with method removal
mod = fp.MethodIntervention(year=2015, label='Remove Withdrawal')
mod.remove_method('wdraw')
intv = mod.build()
removal_sim = fp.Sim(pars=pars, location='senegal', interventions=intv)
removal_sim.run()

# Generate comprehensive visualizations
plot_method_removal_impact(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal',
                          save_path='figures/removal_impact.png')
plot_method_redistribution(baseline_sim, removal_sim, 2015, 'senegal',
                          save_path='figures/redistribution.png')
plot_removed_method_timeline(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal',
                            save_path='figures/timeline.png')
create_removal_summary_figure(baseline_sim, removal_sim, 2000, 2030, 2015, 'senegal',
                             save_path='figures/summary.png')

print("✓ All visualizations saved to figures/")
```

## Visualization Outputs

The plotting functions generate professional, publication-quality figures showing:

### 1. Removal Impact Plot
- **Left panel:** Removed method usage dropping to zero with temporal annotation
- **Right panel:** CPR comparison showing overall contraceptive prevalence impact
- Highlights removal year with vertical line

### 2. Redistribution Analysis
- **Left panel:** Horizontal bar chart showing percent change in each method
- **Right panel:** Stacked bar comparison of baseline vs. post-removal method mix
- Color-coded to show which methods gained users (green) vs. lost users (red)

### 3. Timeline Visualization
- **Top panel:** Baseline stacked area chart (removed method highlighted in red)
- **Bottom panel:** Post-removal stacked area chart (remaining methods only)
- Side-by-side comparison showing method mix evolution

### 4. Summary Figure (5 panels)
- Panel 1: Removed method usage timeline
- Panel 2: CPR comparison over time
- Panel 3: mCPR comparison over time
- Panel 4: Method redistribution bar chart
- Panel 5: Final method mix comparison

All plots are saved at 300 DPI for publication quality.

## Test Results

The implementation was tested with multiple scenarios:

### Test 1: Baseline
- All 10 methods available (including 'none')
- Normal distribution across methods
- ✅ **PASSED**

### Test 2: Implant Removal via Intervention
- Implants removed in 2015
- Users reassigned to Injectables
- Injectable usage increased from ~4% to ~4.5%
- Implants no longer appear in results after 2015
- ✅ **PASSED**

### Test 3: Direct Removal
- IUDs removed at initialization
- Users reassigned to Injectables
- 9 methods remaining (from 10)
- Indices properly re-mapped
- ✅ **PASSED**

### Test 4: Visualization Generation
- All 4 plotting functions tested with multiple scenarios
- Plots correctly identify removed methods
- Handles multiple method removals
- Color coding and annotations work correctly
- ✅ **PASSED**

## Technical Details

### Method Index Re-mapping

When a method is removed:
1. Store removed method's index and name
2. Delete method from methods dict
3. Reassign users currently on that method
4. Re-index remaining methods (decrement indices > removed_idx)
5. Update all method indices in people's state arrays
6. Contract switching matrix
7. Update method mix and method weights arrays

### Switching Matrix Contraction

The switching matrix has multiple nested structures:
- Event types (e.g., 0=normal, 1=postpartum)
- Age bins (e.g., '18-25', '25-35')
- Origin methods (e.g., 'pill', 'inj')
- Destination probabilities and method indices

The `_contract_switching_matrix()` method handles all these structures:
- Removes the method from `method_idx` lists
- Removes probability entries for the removed method
- Shifts remaining indices down
- Renormalizes all probability vectors

### Automatic Visualization Detection

The plotting functions automatically:
- Detect which method(s) were removed by comparing baseline and intervention simulations
- Extract method labels for display
- Calculate redistribution patterns
- Handle multiple simultaneous removals
- Adjust plot layouts based on number of remaining methods

### Safeguards

- Cannot remove the 'none' method (non-use state)
- Validates method exists before removal
- Validates reassignment method exists
- Handles cases where simulation hasn't started yet
- Gracefully handles missing data structures
- Plotting functions handle edge cases (no removals, invalid sims, etc.)

## Known Limitations

1. **Complex Switching Patterns**: Some methods with very complex switching patterns or methods that aren't well-represented in the calibration data may have edge cases
2. **Multiple Sequential Removals**: Removing multiple methods in the same simulation may require additional testing for complex scenarios
3. **Re-adding Methods**: Once removed, methods cannot be automatically re-added (would need a separate `add_method` call)
4. **Visualization Performance**: For very large simulations (>100k agents) or long time periods, plotting may take several seconds

## Files Modified

### Core Functionality
1. **`fpsim/methods.py`**
   - Added `remove_method()` method (~110 lines)
   - Added `_contract_switching_matrix()` helper (~110 lines)

2. **`fpsim/interventions.py`**
   - Added `remove_method` parameter to `update_methods` intervention
   - Updated validation logic
   - Added removal step in intervention execution

3. **`fpsim/wrappers.py`**
   - Added `remove_method()` method to `MethodIntervention` class (~75 lines)
   - Updated `InterventionConfig` dataclass
   - Added `has_remove_method` property
   - Updated `PreviewDict` type definition
   - Updated `build()` method to include remove_method

### Examples and Visualization
4. **`examples/example_remove_method.py`**
   - Comprehensive example file (~234 lines)
   - Demonstrates multiple usage patterns
   - Includes baseline comparison
   - **NEW:** Automatic visualization generation

5. **`examples/example_method_intervention_usage.py`**
   - **NEW:** Added Case 5 demonstrating method removal (~25 lines)
   - **NEW:** Automatic removal plot generation when comparing with case 5 (~30 lines)

6. **`examples/plots.py`**
   - **NEW:** Added `plot_method_removal_impact()` (~100 lines)
   - **NEW:** Added `plot_method_redistribution()` (~105 lines)
   - **NEW:** Added `plot_removed_method_timeline()` (~90 lines)
   - **NEW:** Added `create_removal_summary_figure()` (~180 lines)
   - **NEW:** Added `_get_removed_method_names()` helper function
   - Total: ~475 lines of new visualization code

### Documentation
7. **`examples/PLOTTING_GUIDE.md`**
   - **NEW:** Comprehensive guide to all plotting functions
   - Method addition and removal plot documentation
   - Usage examples and API reference

8. **`MARKDOWNS/REMOVE_METHOD_FEATURE.md`** (this file)
   - Updated with visualization capabilities

9. **`MARKDOWNS/REMOVE_METHOD_QUICKSTART.md`**
   - Updated with visualization examples

## API Summary

### Core Methods

```python
# In ContraceptiveChoice class (methods.py)
def remove_method(self, method_label, reassign_to='none'):
    """Remove a contraceptive method from the simulation."""
    
# In MethodIntervention class (wrappers.py)
def remove_method(self, method: str, reassign_to: str = 'none') -> 'MethodIntervention':
    """Remove a contraceptive method from the simulation (builder pattern)."""
```

### Intervention Parameters

```python
# In update_methods intervention (interventions.py)
fp.update_methods(
    year=2025,
    remove_method={
        'method_label': 'Implants',
        'reassign_to': 'Injectables'
    },
    verbose=True
)
```

### Visualization Functions

```python
# In plots.py
from plots import (
    plot_method_removal_impact,
    plot_method_redistribution,
    plot_removed_method_timeline,
    create_removal_summary_figure
)

# Impact visualization
plot_method_removal_impact(baseline_sim, removal_sim, start_year, end_year, 
                          removal_year, location, save_path='impact.png')

# Redistribution analysis
plot_method_redistribution(baseline_sim, removal_sim, removal_year, location,
                          save_path='redistribution.png')

# Timeline comparison
plot_removed_method_timeline(baseline_sim, removal_sim, start_year, end_year,
                            removal_year, location, save_path='timeline.png')

# Comprehensive summary
create_removal_summary_figure(baseline_sim, removal_sim, start_year, end_year,
                             removal_year, location, save_path='summary.png')
```

## Use Cases

1. **Supply Chain Disruption**: Model what happens when a method becomes temporarily unavailable
2. **Product Discontinuation**: Simulate long-term unavailability of a contraceptive method
3. **Policy Changes**: Model regulatory restrictions on certain methods
4. **Clinical Concerns**: Simulate removal of methods due to safety concerns
5. **Baseline Comparison**: Test interventions with reduced method sets
6. **Impact Analysis**: Visualize and quantify the impact of method unavailability
7. **User Redistribution**: Understand where users go when their method is removed
8. **Publication Figures**: Generate publication-quality visualizations for research papers

## Future Enhancements

Potential improvements for future versions:

### Functionality
1. Support for temporary method removal with automatic re-introduction
2. Batch removal of multiple methods at once
3. Conditional removal based on simulation state
4. More sophisticated reassignment strategies (e.g., proportional to method similarity)
5. Automatic switching matrix recalibration after removal

### Visualization
6. Interactive Plotly versions of removal plots
7. Animation showing method mix evolution over time
8. Comparative plots for multiple removal scenarios
9. Statistical significance testing for method redistribution
10. Integration with FPsim's built-in plotting methods

## Running the Examples

### Basic Example
```bash
cd examples
python example_remove_method.py
```

This will:
- Run 3 different removal scenarios
- Generate comprehensive visualizations
- Save all plots to `figures/` directory

### Integration Example
```bash
cd examples
# Edit example_method_intervention_usage.py to set:
# usecases = [0, 5]
# plot_compare = [[0, 5]]
python example_method_intervention_usage.py
```

This will:
- Compare baseline (case 0) with method removal (case 5)
- Automatically generate specialized removal plots
- Save all visualizations to `figures/` directory

## Conclusion

The `remove_method` functionality provides a comprehensive and user-friendly way to model contraceptive method availability changes in FPsim simulations. It integrates seamlessly with the existing intervention system and provides multiple usage patterns for different user needs.

The new visualization capabilities make it easy to understand and communicate the impact of method removals, with publication-quality figures that show:
- Timeline of method usage before/after removal
- Overall contraceptive prevalence impact (CPR/mCPR)
- User redistribution patterns
- Method mix evolution

Together, these tools enable researchers to model and analyze method availability scenarios with confidence and clarity.
