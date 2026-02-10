# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FPsim is the Institute for Disease Modeling's family planning simulator, an agent-based model for family planning research. It uses a life-course approach to examine how temporal effects unfold over women's lives and how individual-level changes lead to macro-level outcomes.

**Key Architecture:**
- Core simulation engine in `fpsim/sim.py` with `Sim` class
- Agent-based modeling with `People` class in `fpsim/people.py`
- Parameter management via `SimPars`/`FPPars` classes in `fpsim/parameters.py`
- Country-specific calibrations in `fpsim/locations/` directory
- Interventions system for policy modeling in `fpsim/interventions.py`
- Analysis tools in `fpsim/analyzers.py` for extracting results
- Scenario management via `Scenarios` class in `fpsim/scenarios.py`
- Built on top of Starsim >=3.0.2 framework

## Development Commands

### Testing
```bash
# Run all tests with parallel execution
cd tests && ./run_tests

# Run specific test files
pytest test_baselines.py -n auto --durations=0

# Check baseline results
cd tests && python check_baseline

# Check benchmark results  
cd tests && python check_benchmark

# Update baseline (after verifying changes)
cd tests && python update_baseline
```

### Documentation
```bash
# Build documentation
cd docs && ./build_docs

# Build docs without rebuilding notebooks
cd docs && ./build_docs never

# Build docs in debug mode (serial)
cd docs && ./build_docs debug
```

### Installation
```bash
# Development installation
pip install -e .

# Production installation
pip install fpsim
```

## Code Structure

### Core Components
- **Simulation**: `Sim` class runs single simulations, `Scenarios` runs multiple scenarios
- **People**: Agent-based population with demographic and contraceptive states
- **Parameters**: Hierarchical parameter system with location-specific calibrations
- **Methods**: Contraceptive method modeling with switching dynamics
- **Interventions**: Policy interventions that modify simulation parameters
- **Scenarios**: Parameter variations for comparative analysis

### Location System
Country-specific parameters in `fpsim/locations/`:
- Each country has a dedicated module (e.g., `ethiopia/ethiopia.py`)
- Data files in country `data/` subdirectories
- Regional variations supported (e.g., `ethiopia/regions/`)

### Key Dependencies
- Starsim >=3.0.2 (underlying simulation framework)
- NumPy/SciPy for numerical computation
- Pandas for data handling
- Matplotlib/Seaborn for plotting
- Optuna for optimization/calibration

## Testing Architecture

### Test Categories
- `test_baselines.py`: Core functionality regression tests
- `test_dynamics.py`: Population dynamics verification
- `test_interventions.py`: Policy intervention testing
- `test_scenarios.py`: Scenario comparison testing
- `test_multisim.py`: Multi-simulation testing
- `test_sim.py`: Basic simulation testing
- `test_parameters.py`: Parameter system testing
- `test_analyzers.py`: Analyzer testing
- `test_integration.py`: Integration testing
- `test_other.py`: Miscellaneous tests

### Environment Setup
Tests require `SCIRIS_BACKEND=agg` to prevent plot display. This is automatically set in `pytest.ini` and test scripts.

## Development Workflow

1. **Feature Development**: Use `examples/example_scens.py` for quick debugging
2. **Compatibility Testing**: Ensure new features work with both single sims and multisims
3. **Method Compatibility**: Test with novel method introduction scenarios
4. **Unit Testing**: Every new feature requires a corresponding unittest
5. **Style Guide**: Follow starsim style guide (https://github.com/amath-idm/styleguide)

## Common Patterns

### Running Simulations
```python
import fpsim as fp

# Basic simulation
sim = fp.Sim(location='senegal')
sim.run()

# Simulation with custom parameters
sim = fp.Sim(n_agents=10_000, location='kenya', start_year=2000, end_year=2020)
sim.run()
```

### Adding Interventions
```python
# Intervention affects parameter changes over time
intervention = fp.change_par(par='exposure_factor', years=2020, vals=0.5)
sim = fp.Sim(location='senegal', interventions=intervention)
```

### Parameter Access
Parameters use `SimPars` and `FPPars` classes with location-specific defaults. Use `fp.all_pars(location='country')` to get calibrated parameter sets, or pass parameters directly to `fp.Sim()`.