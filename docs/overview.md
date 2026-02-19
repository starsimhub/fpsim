# Overview

FPsim is a stochastic agent-based model, written in Python, for exploring and analyzing family planning. It is designed as an open-source tool for family planning research, using a life-course approach to examine how compounding and temporal effects unfold over women's lives, and how individual-level changes lead to macro-level outcomes.

## Key features

- **Agent-based modeling**: Each woman in the simulation is an individual agent with demographic and contraceptive states.
- **Life-course approach**: Tracks women through their reproductive lives, including fertility, contraceptive use, and life events.
- **Country-specific calibrations**: Includes calibrated parameter sets for multiple countries (Senegal, Kenya, Ethiopia, and more).
- **Contraceptive modeling**: Detailed modeling of contraceptive method choice, switching, and duration of use.
- **Intervention analysis**: Policy interventions that modify simulation parameters to explore "what-if" scenarios.
- **Built on Starsim**: Uses the [Starsim](https://starsim.org) framework (v3.0.2+) for simulation architecture.

## Quick start

```python
import fpsim as fp

# Basic simulation
sim = fp.Sim(location='senegal')
sim.run()

# Simulation with custom parameters
sim = fp.Sim(n_agents=10_000, location='kenya', start_year=2000, end_year=2020)
sim.run()
```
