#!/usr/bin/env python3
"""
Generic script to run a pre-calibrated FPsim model for any location.
This script runs simulations using the calibrated parameters for a location
and generates plots showing how well the model matches data.

Usage:
    python run_calibrated_location.py <location_name>
    python run_calibrated_location.py cotedivoire
    python run_calibrated_location.py nigeria_lagos
    
Options:
    --load      Load existing simulation results instead of running
    --n-agents  Number of agents (default: 5000)
    --end-year  End year for simulation (default: 2020)
    --no-save   Don't save simulation results
    --no-plots  Don't generate plots
"""

import argparse
import numpy as np
import fpsim as fp
import pandas as pd
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from fpsim import plotting as fpplt
import os
import sys
from pathlib import Path

def setup_plotting(location, results_dir='calib_results'):
    """Configure plotting settings"""
    figures_dir = os.path.join(results_dir, 'figures')
    fpplt.Config.set_figs_directory(figures_dir)
    fpplt.Config.do_save = True
    fpplt.Config.do_show = False
    fpplt.Config.show_rmse = True

def plot_calib(sim, single_fig=False, fig_kwargs=None, legend_kwargs=None):
    """Plots the commonly used plots for calibrated model validation
    
    Plotting function which plots the primary validation targets:
    method mix, method use, cpr, total fertility rate, birth spacing, 
    age at first birth, and age-specific fertility rate.
    """
    fpplt.plot_calib(sim)

def run_calibrated_sim(location, n_agents=5000, end_year=2020):
    """Run simulation with pre-calibrated parameters for a given location"""
    print(f"Running pre-calibrated simulation for {location}...")
    
    # Create simulation with parameters - uses calibrated parameters from location files
    pars = dict(
        location=location,
        n_agents=n_agents,
        end_year=end_year,
    )
    
    sim = fp.Sim(pars=pars)
    sim.init()
    sim.run()
    
    return sim

def save_results(sim, location, results_dir='calib_results'):
    """Save simulation results"""
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f'{location}_calib.sim')
    sc.saveobj(filename, sim)
    print(f"Saved simulation results to {filename}")

def load_results(location, results_dir='calib_results'):
    """Load existing simulation results"""
    filename = os.path.join(results_dir, f'{location}_calib.sim')
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No simulation results found at {filename}")
    
    sim = sc.loadobj(filename)
    print(f"Loaded simulation results from {filename}")
    return sim

def main():
    parser = argparse.ArgumentParser(
        description='Run pre-calibrated FPsim simulation for any location',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else ''
    )
    
    parser.add_argument('location', help='Location name (e.g., cotedivoire, nigeria_lagos)')
    parser.add_argument('--load', action='store_true', 
                       help='Load existing simulation results instead of running')
    parser.add_argument('--n-agents', type=int, default=5000,
                       help='Number of agents (default: 5000)')
    parser.add_argument('--end-year', type=int, default=2020,
                       help='End year for simulation (default: 2020)')
    parser.add_argument('--results-dir', default='calib_results',
                       help='Directory for simulation results (default: calib_results)')
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save simulation results")
    parser.add_argument('--no-plots', action='store_true',
                       help="Don't generate validation plots")
    
    args = parser.parse_args()
    
    # Validate location exists
    try:
        # Try to create a sim to validate the location exists
        test_sim = fp.Sim(pars=dict(location=args.location, n_agents=100, test=True))
        test_sim.init()
        print(f"OK Location '{args.location}' is valid")
    except Exception as e:
        print(f"ERROR: Location '{args.location}' is not valid: {e}")
        sys.exit(1)
    
    # Setup plotting
    if not args.no_plots:
        setup_plotting(args.location, args.results_dir)
    
    # Run or load simulation
    if args.load:
        try:
            sim = load_results(args.location, args.results_dir)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        sim = run_calibrated_sim(args.location, args.n_agents, args.end_year)
        
        if not args.no_save:
            save_results(sim, args.location, args.results_dir)
    
    # Generate plots
    if not args.no_plots:
        print("Generating validation plots...")
        sc.options(fontsize=20)  # Set fontsize
        plot_calib(sim, single_fig=True)
        print(f"OK Plots saved to {os.path.join(args.results_dir, 'figures')}")
    
    print(f"OK Pre-calibrated simulation complete for {args.location}")

if __name__ == '__main__':
    main()