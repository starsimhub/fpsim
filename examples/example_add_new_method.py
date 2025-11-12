"""
Showcase: Adding New Contraceptive Methods with Visualizations

This example demonstrates the new capability to dynamically add contraceptive 
methods during a simulation, with comprehensive plots showing:
1. Method mix changes over time
2. CPR/mCPR trends
3. Adoption of new methods
4. Comparison with baseline
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from plots import plot_comparison_full
import fpsim as fp

# Configuration
LOCATION = 'kenya'
N_AGENTS = 10000  # Larger population for clearer trends
START_YEAR = 2000
END_YEAR = 2020
INTERVENTION_YEAR = 2010

def default_pars():
    """Baseline parameters."""
    return dict(
        n_agents=N_AGENTS,
        location=LOCATION,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

def run_baseline():
    """Run baseline simulation without new method."""
    print("Running baseline simulation...")
    pars = default_pars()
    sim = fp.Sim(pars=pars, label='Baseline')
    sim.run()
    return sim

def run_with_new_method():
    """Run simulation with new MY-NEW-METHOD method with comprehensive promotion."""
    print("Running intervention simulation...")
    
    # Define the new method with better baseline properties
    my_new_method = fp.Method(
        name='my_new_method',
        label='MY-NEW-METHOD',
        efficacy=0.80,  # Higher efficacy - well-designed product
        modern=False,
        dur_use=fp.methods.ln(6, 2.5),  # Better baseline duration
        csv_name='MY-NEW-METHOD'
    )
    
    pars = default_pars()
    
    mod = fp.MethodIntervention(year=INTERVENTION_YEAR, label='MY-NEW-METHOD Comprehensive Program')
    
    # Add the method with maximum staying probability
    mod.add_method(
        method=my_new_method,
        copy_from_row='inj',
        copy_from_col='inj',
        initial_share=0.40  # 40% staying probability !
    )
    mod.set_duration_months('my_new_method', 12)  
    mod.set_efficacy('my_new_method', 0.995)
    
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv, label='With MY-NEW-METHOD Program')
    sim.run()
    
    return sim

# region: Plotting


COLORS = {
    'baseline': '#2E86AB',
    'intervention': '#A23B72',
    'new_method': '#F18F01',
}

def plot_injectable_methods_comparison(baseline_sim, intervention_sim, save_path='add_method_injectables.png'):
    """Plot injectable method usage comparison over time."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get years as numeric array
    years = np.linspace(START_YEAR, END_YEAR, len(baseline_sim.results.timevec))
    
    # Get method mix data
    baseline_fp = baseline_sim.connectors['fp']
    intervention_fp = intervention_sim.connectors['fp']
    baseline_mix = baseline_fp.method_mix
    interv_mix = intervention_fp.method_mix
    
    # Get injectable indices
    baseline_methods = baseline_sim.connectors.contraception.methods
    interv_methods = intervention_sim.connectors.contraception.methods
    
    # Left panel: Injectable users over time
    for name in ['inj', 'my_new_method']:
        if name in baseline_methods:
            idx = baseline_methods[name].idx
            label = baseline_methods[name].label
            ax1.plot(years, baseline_mix[idx, :] * 100, 
                    label=f'{label} (Baseline)', linestyle='--', linewidth=2, alpha=0.7)
        
        if name in interv_methods:
            idx = interv_methods[name].idx
            label = interv_methods[name].label
            color = COLORS['new_method'] if name == 'my_new_method' else 'green'
            ax1.plot(years, interv_mix[idx, :] * 100, 
                    label=f'{label} (Program)', linewidth=2.5, color=color)
    
    ax1.axvline(INTERVENTION_YEAR, color='red', linestyle='--', linewidth=2, 
               alpha=0.5, label='Program Start')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Percentage of Users (%)', fontsize=12)
    ax1.set_title('Injectable Methods: Individual Trends', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right panel: Combined injectable share
    baseline_inj_total = baseline_mix[baseline_methods['inj'].idx, :] * 100
    interv_inj_idx = interv_methods['inj'].idx
    interv_my_new_method_idx = interv_methods['my_new_method'].idx
    interv_inj_total = (interv_mix[interv_inj_idx, :] + interv_mix[interv_my_new_method_idx, :]) * 100
    
    ax2.plot(years, baseline_inj_total, label='Baseline (Inj only)', 
            color=COLORS['baseline'], linewidth=2.5)
    ax2.plot(years, interv_inj_total, label='Program (Inj + MY-NEW-METHOD)', 
            color=COLORS['intervention'], linewidth=2.5)
    ax2.axvline(INTERVENTION_YEAR, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add annotation showing increase
    final_increase = interv_inj_total[-1] - baseline_inj_total[-1]
    ax2.text(0.98, 0.95, f'Total injectable\nshare increase:\n+{final_increase:.1f} pp',
            transform=ax2.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.85))
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Percentage of Users (%)', fontsize=12)
    ax2.set_title('Total Injectable Share', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Injectable Methods Comparison in {LOCATION.title()}', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved injectable comparison to {save_path}")
    return fig


def plot_method_mix_evolution(sim, title='Method Mix Over Time', save_path='add_method_mix.png'):
    """Plot stacked area chart of method mix over time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get method mix data from FPmod
    fp_mod = sim.connectors['fp']
    method_mix = fp_mod.method_mix
    # Get years as numeric array
    years = np.linspace(START_YEAR, END_YEAR, len(sim.results.timevec))
    
    # Get method labels
    methods = sim.connectors.contraception.methods
    labels = [m.label for m in methods.values() if m.name != 'none']
    
    # Exclude 'none' method from visualization
    method_mix_users = method_mix[1:, :]  # Skip index 0 which is 'none'
    
    # Create color palette
    n_methods = len(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_methods))
    
    # Plot stacked area
    ax.stackplot(years, method_mix_users, labels=labels, colors=colors, alpha=0.8)
    
    ax.axvline(INTERVENTION_YEAR, color='red', linestyle='--', linewidth=2, 
               label='MY-NEW-METHOD Introduced', alpha=0.7)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion of Users', fontsize=12)
    ax.set_title(f'{title} in {LOCATION.title()}', fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add note about MY-NEW-METHOD (if it exists)
    if len(labels) > 9:  # More than standard 9 methods
        ax.text(0.02, 0.02, 'Note: MY-NEW-METHOD band appears at top after 2010 (cyan/light blue)',
               transform=ax.transAxes, fontsize=9, style='italic',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved method mix evolution to {save_path}")
    return fig


def plot_new_method_adoption(sim, save_path='add_method_adoption.png'):
    """Plot adoption of the new method over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get method mix data
    fp_mod = sim.connectors['fp']
    method_mix = fp_mod.method_mix
    # Get years as numeric array
    years = np.linspace(START_YEAR, END_YEAR, len(sim.results.timevec))
    
    # Find MY-NEW-METHOD index
    methods = sim.connectors.contraception.methods
    my_new_method_idx = None
    for name, method in methods.items():
        if name == 'my_new_method':
            my_new_method_idx = method.idx
            break
    
    if my_new_method_idx is not None:
        # Plot MY-NEW-METHOD adoption
        my_new_method_proportion = method_mix[my_new_method_idx, :]
        ax.fill_between(years, 0, my_new_method_proportion * 100, 
                        color=COLORS['new_method'], alpha=0.6, label='MY-NEW-METHOD Users')
        ax.plot(years, my_new_method_proportion * 100, 
                color=COLORS['new_method'], linewidth=3, label='MY-NEW-METHOD Trend')
        
        # Mark intervention point
        ax.axvline(INTERVENTION_YEAR, color='red', linestyle='--', linewidth=2, 
                  label='Introduction Year', alpha=0.7)
        
        # Add annotations
        final_adoption = my_new_method_proportion[-1] * 100
        if final_adoption > 0.1:
            ax.annotate(f'Final adoption:\n{final_adoption:.2f}%', 
                       xy=(END_YEAR, my_new_method_proportion[-1] * 100),
                       xytext=(END_YEAR - 3, final_adoption + 2),
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add information about base method
    info_text = ('MY-NEW-METHOD Program Components:\n'
                '• Base patterns: Injectables\n'
                '• Staying probability: 60%\n'
                '• Duration: 60 months (5 years!)\n'
                '• Efficacy: 99.5%\n'
                '• Comprehensive injectable/LARC program')
    ax.text(0.02, 0.97, info_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.85, edgecolor='darkgreen', linewidth=2))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentage of Users (%)', fontsize=12)
    ax.set_title(f'MY-NEW-METHOD Adoption in {LOCATION.title()} - Comprehensive Program Impact', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([START_YEAR, END_YEAR])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved new method adoption to {save_path}")
    return fig


def plot_method_comparison_bar(baseline_sim, intervention_sim, save_path='add_method_bar.png'):
    """Bar chart comparing final method usage."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get final method usage for both sims
    baseline_methods = {}
    intervention_methods = {}
    
    # Baseline
    ppl = baseline_sim.people
    methods = baseline_sim.connectors.contraception.methods
    for name, method in methods.items():
        if name != 'none':
            count = np.sum(ppl.fp.method == method.idx)
            baseline_methods[method.label] = count
    
    # Intervention
    ppl = intervention_sim.people
    methods = intervention_sim.connectors.contraception.methods
    for name, method in methods.items():
        if name != 'none':
            count = np.sum(ppl.fp.method == method.idx)
            intervention_methods[method.label] = count
    
    # Prepare data for plotting
    all_methods = sorted(set(list(baseline_methods.keys()) + list(intervention_methods.keys())))
    baseline_counts = [baseline_methods.get(m, 0) for m in all_methods]
    intervention_counts = [intervention_methods.get(m, 0) for m in all_methods]
    
    # Create bar chart
    x = np.arange(len(all_methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_counts, width, label='Baseline', 
                   color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar(x + width/2, intervention_counts, width, label='With MY-NEW-METHOD', 
                   color=COLORS['intervention'], alpha=0.8)
    
    # Highlight new method
    if 'MY-NEW-METHOD' in all_methods:
        new_idx = all_methods.index('MY-NEW-METHOD')
        bars2[new_idx].set_color(COLORS['new_method'])
        bars2[new_idx].set_edgecolor('black')
        bars2[new_idx].set_linewidth(2)
    
    ax.set_xlabel('Contraceptive Method', fontsize=12)
    ax.set_ylabel('Number of Users', fontsize=12)
    ax.set_title(f'Contraceptive Method Usage at End of Simulation ({END_YEAR})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved method comparison bar chart to {save_path}")
    return fig


def plot_births_comparison(baseline_sim, intervention_sim, save_path='add_method_births.png'):
    """Plot births over time comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get years as numeric array
    years = np.linspace(START_YEAR, END_YEAR, len(baseline_sim.results.timevec))
    
    # Get births
    baseline_births = baseline_sim.results.fp.births
    intervention_births = intervention_sim.results.fp.births
    
    # Plot
    ax.plot(years, baseline_births, label='Baseline', 
           color=COLORS['baseline'], linewidth=2, marker='o', markersize=3)
    ax.plot(years, intervention_births, label='With MY-NEW-METHOD', 
           color=COLORS['intervention'], linewidth=2, marker='s', markersize=3)
    ax.axvline(INTERVENTION_YEAR, color='gray', linestyle='--', alpha=0.5, label='Intervention')
    
    # Calculate and show cumulative difference
    years_mask = years >= INTERVENTION_YEAR
    baseline_total = np.sum(baseline_births[years_mask])
    intervention_total = np.sum(intervention_births[years_mask])
    births_averted = baseline_total - intervention_total
    
    ax.text(0.02, 0.98, f'Births after intervention:\nBaseline: {int(baseline_total)}\nWith MY-NEW-METHOD: {int(intervention_total)}\nDifference: {int(births_averted)}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Births per Month', fontsize=12)
    ax.set_title(f'Births Over Time: Baseline vs. MY-NEW-METHOD Program in {LOCATION.title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved births comparison to {save_path}")
    return fig


def create_summary_figure(baseline_sim, intervention_sim, save_path='add_method_summary.png'):
    """Create a comprehensive summary figure with multiple subplots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Get years as numeric array
    years = np.linspace(START_YEAR, END_YEAR, len(baseline_sim.results.timevec))
    
    # Get method mix data
    baseline_fp = baseline_sim.connectors['fp']
    intervention_fp = intervention_sim.connectors['fp']
    baseline_mix = baseline_fp.method_mix
    interv_mix = intervention_fp.method_mix
    baseline_methods = baseline_sim.connectors.contraception.methods
    interv_methods = intervention_sim.connectors.contraception.methods
    
    # 1. MY-NEW-METHOD Adoption (main feature!)
    ax1 = fig.add_subplot(gs[0, :])
    my_new_method_idx = interv_methods['my_new_method'].idx
    my_new_method_proportion = interv_mix[my_new_method_idx, :] * 100
    ax1.fill_between(years, 0, my_new_method_proportion, color=COLORS['new_method'], alpha=0.4)
    ax1.plot(years, my_new_method_proportion, color=COLORS['new_method'], linewidth=3, label='MY-NEW-METHOD Users')
    ax1.axvline(INTERVENTION_YEAR, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Introduction')
    
    # Add program info
    info_text = 'Program: Inj patterns | 60% staying | 60 mo duration | 99.5% efficacy | LARC-level continuation'
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', alpha=0.85, edgecolor='darkgreen'))
    
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('% of Users', fontsize=11)
    ax1.set_title('MY-NEW-METHOD Adoption Rate', fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Injectable Methods Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    baseline_inj = baseline_mix[baseline_methods['inj'].idx, :] * 100
    interv_inj = interv_mix[interv_methods['inj'].idx, :] * 100
    interv_my_new_method = interv_mix[interv_methods['my_new_method'].idx, :] * 100
    
    ax2.plot(years, baseline_inj, label='Regular Inj (Baseline)', 
            color=COLORS['baseline'], linewidth=2, linestyle='--', alpha=0.7)
    ax2.plot(years, interv_inj, label='Regular Inj (Program)', 
            color='green', linewidth=2.5)
    ax2.plot(years, interv_my_new_method, label='MY-NEW-METHOD (New)', 
            color=COLORS['new_method'], linewidth=2.5)
    ax2.axvline(INTERVENTION_YEAR, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('% of Users', fontsize=11)
    ax2.set_title('Injectable Methods: Individual Trends', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Total Injectable Share
    ax3 = fig.add_subplot(gs[1, 1])
    baseline_inj_total = baseline_inj
    interv_inj_total = interv_inj + interv_my_new_method
    
    ax3.fill_between(years, baseline_inj_total, interv_inj_total, 
                    where=(interv_inj_total >= baseline_inj_total),
                    color='green', alpha=0.3, label='Increase')
    ax3.plot(years, baseline_inj_total, label='Baseline Total', 
            color=COLORS['baseline'], linewidth=2.5)
    ax3.plot(years, interv_inj_total, label='Program Total', 
            color=COLORS['intervention'], linewidth=2.5)
    ax3.axvline(INTERVENTION_YEAR, color='red', linestyle='--', alpha=0.5)
    
    final_increase = interv_inj_total[-1] - baseline_inj_total[-1]
    ax3.text(0.5, 0.95, f'+{final_increase:.1f} pp increase',
            transform=ax3.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    ax3.set_xlabel('Year', fontsize=11)
    ax3.set_ylabel('% of Users', fontsize=11)
    ax3.set_title('Total Injectable Share (Combined)', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Method Substitution (which methods decreased?)
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Calculate changes in method usage
    method_changes = {}
    all_method_names = set()
    
    for name in baseline_methods.keys():
        all_method_names.add(baseline_methods[name].label)
    for name in interv_methods.keys():
        all_method_names.add(interv_methods[name].label)
    
    for label in all_method_names:
        baseline_pct = 0
        interv_pct = 0
        
        for name, method in baseline_methods.items():
            if method.label == label and name != 'none':
                baseline_pct = baseline_mix[method.idx, -1] * 100
                
        for name, method in interv_methods.items():
            if method.label == label and name != 'none':
                interv_pct = interv_mix[method.idx, -1] * 100
        
        change = interv_pct - baseline_pct
        if abs(change) > 0.01:  # Only show meaningful changes
            method_changes[label] = change
    
    # Sort by change
    sorted_methods = sorted(method_changes.items(), key=lambda x: x[1], reverse=True)
    labels = [m[0] for m in sorted_methods]
    changes = [m[1] for m in sorted_methods]
    
    colors_bars = [COLORS['new_method'] if c > 0 else 'lightcoral' for c in changes]
    bars = ax4.barh(labels, changes, color=colors_bars, alpha=0.8, edgecolor='black')
    ax4.axvline(0, color='black', linewidth=1)
    ax4.set_xlabel('Change in % of Users', fontsize=11)
    ax4.set_title('Method Substitution Effects', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (label, val) in enumerate(zip(labels, changes)):
        ax4.text(val + (0.3 if val > 0 else -0.3), i, f'{val:+.1f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9, fontweight='bold')
    
    # 5. Top Methods Ranking
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Get top 6 methods by intervention usage
    ppl = intervention_sim.people
    method_counts = []
    for name, method in interv_methods.items():
        if name != 'none':
            count = np.sum(ppl.fp.method == method.idx)
            method_counts.append((method.label, count, name))
    
    method_counts.sort(key=lambda x: x[1], reverse=True)
    top_methods = method_counts[:6]
    
    top_labels = [m[0] for m in top_methods]
    top_counts = [m[1] for m in top_methods]
    top_names = [m[2] for m in top_methods]
    
    colors_top = [COLORS['new_method'] if n == 'my_new_method' else COLORS['intervention'] for n in top_names]
    bars = ax5.barh(top_labels, top_counts, color=colors_top, alpha=0.8, edgecolor='black')
    
    # Highlight MY-NEW-METHOD
    if 'my_new_method' in top_names:
        sc_idx = top_names.index('my_new_method')
        bars[sc_idx].set_linewidth(3)
    
    ax5.set_xlabel('Number of Users', fontsize=11)
    ax5.set_title(f'Top 6 Methods by Usage ({END_YEAR})', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, count in enumerate(top_counts):
        ax5.text(count + 20, i, f'{int(count)}', va='center', fontsize=9, fontweight='bold')
    
    # 6. All Methods Comparison
    ax6 = fig.add_subplot(gs[3, :])
    
    # Get all methods and their final usage
    baseline_method_data = {}
    intervention_method_data = {}
    
    ppl = baseline_sim.people
    for name, method in baseline_methods.items():
        if name != 'none':
            count = np.sum(ppl.fp.method == method.idx)
            baseline_method_data[method.label] = count
    
    ppl = intervention_sim.people
    for name, method in interv_methods.items():
        if name != 'none':
            count = np.sum(ppl.fp.method == method.idx)
            intervention_method_data[method.label] = count
    
    all_methods = sorted(set(list(baseline_method_data.keys()) + list(intervention_method_data.keys())))
    baseline_counts = [baseline_method_data.get(m, 0) for m in all_methods]
    intervention_counts = [intervention_method_data.get(m, 0) for m in all_methods]
    
    x = np.arange(len(all_methods))
    width = 0.35
    ax6.bar(x - width/2, baseline_counts, width, label='Baseline', 
           color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars = ax6.bar(x + width/2, intervention_counts, width, label='With MY-NEW-METHOD Program', 
                   color=COLORS['intervention'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight MY-NEW-METHOD
    if 'MY-NEW-METHOD' in all_methods:
        new_idx = all_methods.index('MY-NEW-METHOD')
        bars[new_idx].set_color(COLORS['new_method'])
        bars[new_idx].set_edgecolor('black')
        bars[new_idx].set_linewidth(3)
    
    ax6.set_xlabel('Contraceptive Method', fontsize=11)
    ax6.set_ylabel('Number of Users', fontsize=11)
    ax6.set_title(f'All Methods: Final Usage Comparison ({END_YEAR})', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'Impact of Introducing MY-NEW-METHOD in {LOCATION.title()} (2010-2020) - Comprehensive Program', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary figure to {save_path}")
    return fig


def print_summary_statistics(baseline_sim, intervention_sim):
    """Print summary statistics comparing both simulations."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Final CPR/mCPR
    print(f"\nFinal Prevalence Rates ({END_YEAR}):")
    print(f"{'Metric':<20} {'Baseline':>15} {'With MY-NEW-METHOD':>15} {'Change':>15}")
    print("-" * 70)
    
    baseline_mcpr = baseline_sim.results.contraception.mcpr[-1]
    interv_mcpr = intervention_sim.results.contraception.mcpr[-1]
    print(f"{'mCPR':<20} {baseline_mcpr:>14.3f} {interv_mcpr:>14.3f} {interv_mcpr-baseline_mcpr:>+14.3f}")
    
    baseline_cpr = baseline_sim.results.contraception.cpr[-1]
    interv_cpr = intervention_sim.results.contraception.cpr[-1]
    print(f"{'CPR':<20} {baseline_cpr:>14.3f} {interv_cpr:>14.3f} {interv_cpr-baseline_cpr:>+14.3f}")
    
    # Total births
    years_numeric = np.linspace(START_YEAR, END_YEAR, len(baseline_sim.results.timevec))
    years_mask = years_numeric >= INTERVENTION_YEAR
    baseline_births = np.sum(baseline_sim.results.fp.births[years_mask])
    interv_births = np.sum(intervention_sim.results.fp.births[years_mask])
    births_averted = baseline_births - interv_births
    
    print(f"\nBirths After Intervention ({INTERVENTION_YEAR}-{END_YEAR}):")
    print(f"{'Baseline births':<30} {int(baseline_births):>10,}")
    print(f"{'With MY-NEW-METHOD births':<30} {int(interv_births):>10,}")
    print(f"{'Births averted':<30} {int(births_averted):>10,}")
    print(f"{'Percent reduction':<30} {100*births_averted/baseline_births:>9.1f}%")
    
    # MY-NEW-METHOD adoption
    fp_mod = intervention_sim.connectors['fp']
    method_mix = fp_mod.method_mix
    methods = intervention_sim.connectors.contraception.methods
    
    for name, method in methods.items():
        if name == 'my_new_method':
            final_adoption = method_mix[method.idx, -1] * 100
            ppl = intervention_sim.people
            n_users = np.sum(ppl.fp.method == method.idx)
            print(f"\nMY-NEW-METHOD Adoption:")
            print(f"{'Final adoption rate':<30} {final_adoption:>9.2f}%")
            print(f"{'Number of users':<30} {n_users:>10,}")
            break

# endregion
if __name__ == '__main__':
    import os
    from pathlib import Path
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("SHOWCASE: Adding New Contraceptive Methods to FPsim")
    print("="*70)
    print(f"\nSimulation Setup:")
    print(f"  Location: {LOCATION.title()} Population: {N_AGENTS:,} agents")
    print(f"  Period: {START_YEAR}-{END_YEAR}")
    print(f"  Intervention: Comprehensive MY-NEW-METHOD Program ({INTERVENTION_YEAR})")
    print(f"  Figures will be saved to: {figures_dir}")
    
    # Run simulations
    baseline_sim = run_baseline()
    intervention_sim = run_with_new_method()
    plot_comparison_full(baseline_sim, intervention_sim)    

    # Create all plots - focus on method-specific metrics
    plot_injectable_methods_comparison(baseline_sim, intervention_sim, 
                                      save_path=figures_dir / 'add_method_injectables.png')
    plot_method_mix_evolution(intervention_sim, 'Method Mix Evolution with MY-NEW-METHOD',
                             save_path=figures_dir / 'add_method_mix.png')
    plot_new_method_adoption(intervention_sim,
                            save_path=figures_dir / 'add_method_adoption.png')
    plot_method_comparison_bar(baseline_sim, intervention_sim,
                              save_path=figures_dir / 'add_method_bar.png')
    plot_births_comparison(baseline_sim, intervention_sim,
                          save_path=figures_dir / 'add_method_births.png')
    create_summary_figure(baseline_sim, intervention_sim,
                         save_path=figures_dir / 'add_method_summary.png')
    
    # Print statistics
    print_summary_statistics(baseline_sim, intervention_sim)
    
    print(f"\n{'='*70}")
    print(f"All figures saved to: {figures_dir}")
    print(f"{'='*70}")
    

