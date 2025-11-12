import numpy as np
import matplotlib.pyplot as plt
from fpsim.sim import Sim
from fpsim.interventions import update_methods
import sciris as sc
import starsim as ss
import fpsim as fp
from fpsim import defaults as fpd
import numpy as np
import plotly.graph_objects as go
from enum import Enum
import os
from pathlib import Path

__all__ = ['plot_method_mix_simple', 'plot_comparison_full', 'plot_method_bincount'
           'plot_method_mix_by_type_per_year', 'iterable_plots', 'plot_pregnancies_per_year',
           'plot_injectable_methods_comparison', 'plot_method_mix_evolution', 'plot_new_method_adoption',
           'plot_method_comparison_bar', 'plot_births_comparison', 'create_summary_figure',
           'print_summary_statistics']

# region: Helper functions for plotting and data extraction
def _get_results_dir():
    """
    Get the results directory path, creating it if it doesn't exist.
    The results folder is created in the same location as the running script.
    
    Returns:
    --------
    Path
        Path object pointing to the results directory
    """
    # Get the directory of the running script (or current working directory as fallback)
    script_dir = Path(os.getcwd())
    results_dir = script_dir / 'figures'
    
    # Create the directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)
    
    return results_dir


class MethoType(Enum):
    NONE = 'None'
    PILL = 'Pill'
    IUD = 'IUD'
    INJECTABLE = 'Injectable'
    CONDOM = 'Condom'
    BTL = 'BTL'
    WITHDRAWAL = 'Withdrawal'
    IMPLANT = 'Implant'
    OTHER_TRADITIONAL = 'Other traditional'
    OTHER_MODERN = 'Other modern'

def _process_colors(color_spec, num_items, default_cmap='Set3'):
    """
    Sample usage:
    >>> _process_colors(None, 5)  # Uses Set3 colormap
    >>> _process_colors('Blues', 5)  # Generates 5 colors from Blues
    >>> _process_colors(['red', 'blue'], 5)  # ['red', 'blue', 'red', 'blue', 'red']
    """
    if color_spec is None:
        return plt.colormaps.get(default_cmap)(np.linspace(0, 1, num_items))
    elif isinstance(color_spec, str):
        # Check if it's a valid colormap name
        if color_spec in plt.colormaps:
            return plt.colormaps.get(color_spec)(np.linspace(0, 1, num_items))
        else:
            # Single color string - repeat it
            return [color_spec] * num_items
    elif isinstance(color_spec, list):
        # Use provided colors, repeating if necessary
        if len(color_spec) < num_items:
            # Repeat colors if we don't have enough
            repeat_times = (num_items // len(color_spec)) + 1
            color_spec = (color_spec * repeat_times)[:num_items]
        return color_spec[:num_items]
    else:
        # Single color - repeat it
        return [color_spec] * num_items


def _get_default_colors():
    """
    Get default color scheme for plots.
    
    Returns:
    --------
    dict
        Dictionary with default colors for all plot elements
    """
    return {
        'baseline': 'purple',
        'intervention': 'yellow',
        'pie': 'viridis',
        'heatmap': 'viridis',
        'difference': None,  # Will use default matplotlib colors
        'baseline_pie': 'viridis',
        'intervention_pie': 'viridis',
    }


def _process_plot_colors(colors=None):
    """
    Parameters:
    -----------
    colors : dict or None
        User-provided colors dict. Can contain keys:
        - 'baseline': color for baseline bars/lines
        - 'intervention': color for intervention bars/lines
        - 'pie': colormap for pie charts
        - 'heatmap': colormap for heatmaps
        - 'difference': color for difference bars
        - 'baseline_pie': colormap for baseline pie chart
        - 'intervention_pie': colormap for intervention pie chart
        
    Returns: dict - Complete colors dict with defaults filled in
    """
    defaults = _get_default_colors()
    if colors is None:
        return defaults
    # Merge user colors with defaults
    result = defaults.copy()
    result.update(colors)
    return result


def _get_data(*sims, exclude_none=False):
    # Extract method counts, normalized proportions, and labels from one or more simulations.
    # Uses method_mix from FPmod connector which includes time series data for all methods,
    # including dynamically added ones.
    # Returns: tuple (method_labels, *method_mix_arrays) where method_mix_arrays are normalized proportions
    # for each simulation, in the same order as input sims
    # Standard method index to name mapping
    
    # Method indices must match the order in fpsim.methods.make_method_list()
    METHOD_INDEX_TO_NAME = {
        0: 'None',
        1: 'Pill',
        2: 'IUDs',
        3: 'Injectables',
        4: 'Condoms',
        5: 'BTL',
        6: 'Withdrawal',
        7: 'Implants',
        8: 'Other traditional',
        9: 'Other modern'
    }
    
    if len(sims) == 0:
        return [], []
    
    # Get max method index across all simulations by checking the contraception connector
    max_method_idx = 0
    for sim in sims:
        try:
            cm = sim.connectors.contraception
            n_methods = len(cm.methods)
            max_method_idx = max(max_method_idx, n_methods)
        except (AttributeError, KeyError):
            # Fall back to people array if connector not available
            if hasattr(sim, 'people') and hasattr(sim.people, 'fp'):
                method_array = sim.people.fp.method
                max_method_idx = max(max_method_idx, int(method_array.max()) + 1 if len(method_array) > 0 else 0)
    
    # Extract method_mix from FPmod connector (includes all methods, even dynamically added)
    # Use final timestep as the snapshot for comparison
    method_mix_arrays = []
    for sim in sims:
        try:
            fp_mod = sim.connectors['fp']
            method_mix_timeseries = fp_mod.method_mix  # Shape: (n_methods, n_timesteps)
            # Use the final timestep
            final_mix = method_mix_timeseries[:, -1]
            # Ensure it has the right length (pad with zeros if needed)
            if len(final_mix) < max_method_idx:
                padded = np.zeros(max_method_idx)
                padded[:len(final_mix)] = final_mix
                final_mix = padded
            method_mix_arrays.append(final_mix)
        except (AttributeError, KeyError):
            # Fall back to people array counting
            method_array = sim.people.fp.method
            counts = np.bincount(method_array, minlength=max_method_idx)
            total = np.sum(counts)
            mix = (counts / total) if total > 0 else counts
            method_mix_arrays.append(np.array(mix))
    
    # Create method labels - try to get from simulation first (for dynamically added methods)
    # Then fall back to standard mapping
    method_labels = []
    for i in range(max_method_idx):
        label = None
        # Try to get label from any simulation's contraception module
        for sim in sims:
            try:
                cm = sim.connectors.contraception
                for method_name, method_obj in cm.methods.items():
                    if method_obj.idx == i:
                        # Use the label if available, otherwise use the capitalized name
                        label = method_obj.label if method_obj.label else method_name.replace('_', '-').upper()
                        break
                if label:
                    break
            except (AttributeError, KeyError):
                continue
        
        # Fall back to standard mapping if not found in simulation
        if not label:
            label = METHOD_INDEX_TO_NAME.get(i, f'Method_{i}')
        
        method_labels.append(label)
    method_indices = np.arange(max_method_idx)
    if exclude_none:
        # Exclude the 'none' method (index 0) but DON'T renormalize
        # The method_mix values already represent proportions of all women,
        # so we just remove 'none' without adjusting the other values
        none_idx = 0
        mask = method_indices != none_idx
        method_indices = method_indices[mask]
        method_labels = [method_labels[i] for i in method_indices]
        
        # Filter all method mix arrays (maintains index order, NO renormalization)
        filtered_arrays = []
        for mix_array in method_mix_arrays:
            filtered = mix_array[mask]
            filtered_arrays.append(filtered)
        method_mix_arrays = filtered_arrays
    else:
        # Keep 'none' included; keep labels and proportions as-is
        pass
    
    # Data is sorted by method index (excluding 'none' when exclude_none=True)
    # - method_labels[i] corresponds to method index method_indices[i]
    # - method_mix_arrays[j][i] corresponds to method index method_indices[i] for simulation j
    # - Indices are in ascending order: [1, 2, 3, ...] if exclude_none=True, or [0, 1, 2, ...] if False
    
    return (method_labels,) + tuple(method_mix_arrays)


def _add_annotations(fig, interventions):
        # Add annotations for methods added via copy_from_existing
    if interventions is not None:
        intervention_events = []
        for intv in interventions:
            # Check if this is a copy_from_existing intervention
            if hasattr(intv, 'name') and 'copy_from_existing' in str(intv.name):
                if hasattr(intv, 'pars') and hasattr(intv.pars, 'new_method') and intv.pars.new_method is not None:
                    method = intv.pars.new_method
                    year = intv.pars.year
                    intervention_events.append({
                        'year': year,
                        'method_label': method.label,
                        'method_name': method.name
                    })
        
        # Add vertical lines and annotations for each copy_from_existing event
        if intervention_events:
            # Get y-range from the data
            all_y_values = []
            for i in range(0, 10):
                all_y_values.extend(sim.connectors.fp.method_mix[i])
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max - y_min
            
            for event_index, event in enumerate(intervention_events):
                year = event['year']
                method_label = event['method_label']
                
                # Add vertical line
                fig.add_vline(
                    x=year,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    line_width=1
                )
                
                # Add text annotation
                # Position annotation at different heights to avoid overlap (from top of plot)
                annotation_offset = (event_index % 3 + 1) * (y_range * 0.08)
                y_pos = y_max - annotation_offset
                
                fig.add_annotation(
                    x=year,y=y_pos,
                    text=f'+ {method_label}',
                    showarrow=True,
                    arrowhead=1, arrowwidth=1, arrowcolor="gray",
                    ax=10, ay=0,
                    bgcolor="lightblue",
                    bordercolor="lightblue",
                    borderwidth=1,
                    font=dict(size=9, color="black"),
                    opacity=0.7
                )
    return fig
# endregion: Helper functions
# region: Plotting functions for METHOD MIX COMPARISON
def plot_method_mix_simple(sim_baseline, sim_intervention, pie_sim, title="Method Mix Comparison", chart_titles=None, colors=None, exclude_none=True):
    """
    Create a simple, clean figure showing method mix comparison.
    
    Parameters:
    -----------
    sim_baseline : Sim
        Baseline simulation object
    sim_intervention : Sim  
        Intervention simulation object
    pie_sim : Sim
        Simulation object for pie chart
    title : str
        Title for the figure
    chart_titles : dict
        Dictionary of titles for individual charts
    colors : dict or None
        Dictionary with custom colors for different elements. Keys can be:
        'baseline', 'intervention', 'pie'. Values can be:
        - matplotlib colormap name (str)
        - list of color names/hex codes
        - color name or hex code (single color)
        If None, uses default colors.
    exclude_none : bool, default False
        If True, exclude the 'none' method (typically index 0) from the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    
    # Extract method counts and labels using shared helper (no connector usage)
    method_labels, baseline_mix, intervention_mix, pie_mix = _get_data(
        sim_baseline, sim_intervention, pie_sim, exclude_none=exclude_none
    )
    
    # Set default chart titles if not provided
    if chart_titles is None:
        chart_titles = {
            'bar_chart': 'Method Mix Comparison',
            'pie_chart': pie_sim.label
        }
    
    # Create simple figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Process colors with defaults
    plot_colors = _process_plot_colors(colors)
    baseline_color = plot_colors['baseline']
    intervention_color = plot_colors['intervention']
    pie_colors = plot_colors.get('pie', 'viridis')
    
    # Get edge colors if specified (backward compatibility)
    if colors is not None:
        baseline_edge = colors.get('baseline_edge', 'black')
        intervention_edge = colors.get('intervention_edge', 'black')
    else:
        baseline_edge = 'black'
        intervention_edge = 'black'
    
    # Left plot: Side-by-side bars
    x = np.arange(len(method_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_mix * 100, width, label='Baseline', 
                   color=baseline_color, alpha=0.8, edgecolor=baseline_edge, linewidth=1)
    
    bars2 = ax1.bar(x + width/2, intervention_mix * 100, width, label='Intervention', 
                   color=intervention_color, alpha=0.8, edgecolor=intervention_edge, linewidth=1)
    
    ax1.set_xlabel('Contraceptive Methods', fontsize=8)
    ax1.set_ylabel('Method Mix (%)', fontsize=8)
    ax1.set_title(chart_titles['bar_chart'], fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Pie chart
    pie_colors_list = _process_colors(pie_colors, len(method_labels))
    
    ax2.pie(pie_mix * 100, labels=method_labels, autopct='%1.1f%%', textprops={'fontsize': 5},
            startangle=90, colors=pie_colors_list)
    ax2.set_title(chart_titles['pie_chart'], fontsize=9)
    # Reduce pad in pie chart
    ax2.title.set_position([0.5, 1.05])
    
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    
    return fig

# endregion: Plotting functions for METHOD MIX COMPARISON
# region: Plotting functions for METHOD MIX COMPARISON FULL
def plot_comparison_full(sim_baseline, 
                         sim_intervention, 
                         show_figure=True, 
                         save_figure=False, 
                         filename="method_mix_comparison.png", 
                         title="Method Mix Matrix Analysis: Intervention Impact Visualization", 
                         chart_titles=None, colors=None, exclude_none=True):
    """
    Create comprehensive visualization of method mix matrix changes.
    
    Parameters:
    -----------
    sim_baseline : Sim
        Baseline simulation object
    sim_intervention : Sim  
        Intervention simulation object
    show_figure : bool
        Whether to display the figure
    save_figure : bool
        Whether to save the figure to file
    filename : str
        Filename for saved figure
    title : str
        Main title for the entire figure
    chart_titles : dict
        Dictionary of titles for individual charts
    colors : dict or None
        Dictionary with custom colors for different elements. Keys can be:
        - 'baseline': color for baseline bars (default: 'purple')
        - 'intervention': color for intervention bars (default: 'yellow')
        - 'pie': colormap for all pie charts (default: 'viridis')
        - 'baseline_pie': colormap for baseline pie chart (default: 'viridis')
        - 'intervention_pie': colormap for intervention pie chart (default: 'viridis')
        - 'heatmap': colormap for heatmap (default: 'viridis')
        - 'difference': color for difference bars (default: None, uses matplotlib default)
        Values can be matplotlib colormap names (str), lists of colors,
        or single color names/hex codes. If None, uses default colors.
        
        Example:
        >>> colors = {
        ...     'baseline': 'steelblue',
        ...     'intervention': 'crimson',
        ...     'pie': 'plasma',
        ...     'heatmap': 'coolwarm'
        ... }
    exclude_none : bool, default False
        If True, exclude the 'none' method (typically index 0) from the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Extract method counts and labels using shared helper (no connector usage)
    method_labels, baseline_mix, intervention_mix = _get_data(
        sim_baseline, sim_intervention, exclude_none=exclude_none
    )
    
    # Calculate differences
    differences = intervention_mix - baseline_mix
    
    # Set default chart titles if not provided
    if chart_titles is None:
        chart_titles = {
            'main_comparison': 'Method Mix Comparison: Baseline vs Intervention',
            'method_changes': 'Method Mix Changes',
            'baseline_pie': 'Baseline Method Mix',
            'intervention_pie': 'Intervention Method Mix',
            'heatmap': 'Method Mix Heatmap: Baseline vs Intervention vs Changes'
        }
    
    # Process colors with defaults
    plot_colors = _process_plot_colors(colors)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.5, height_ratios=[1.0, 1.0, 0.7])
    
    
    # =============================================================================
    # TOP PANEL: Side-by-side comparison bars
    # =============================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(method_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_mix * 100, width, label='Baseline', 
                   color=plot_colors['baseline'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, intervention_mix * 100, width, label='Intervention', 
                   color=plot_colors['intervention'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars using vectorized operations
    heights1 = np.array([bar.get_height() for bar in bars1])
    heights2 = np.array([bar.get_height() for bar in bars2])
    x_positions1 = np.array([bar.get_x() + bar.get_width()/2. for bar in bars1])
    x_positions2 = np.array([bar.get_x() + bar.get_width()/2. for bar in bars2])
    
    # Vectorized label application
    for idx in range(len(bars1)):
        ax1.text(x_positions1[idx], heights1[idx] + 0.5, f'{heights1[idx]:.1f}%',
                ha='center', va='bottom', fontsize=7)
        ax1.text(x_positions2[idx], heights2[idx] + 0.5, f'{heights2[idx]:.1f}%',
                ha='center', va='bottom', fontsize=7)
    
    ax1.set_xlabel('Contraceptive Methods', fontsize=8)
    ax1.set_ylabel('Method Mix Percentage (%)', fontsize=8)
    ax1.set_title(chart_titles['main_comparison'], fontsize=9, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =============================================================================
    # MIDDLE LEFT: Difference Analysis
    # =============================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create horizontal bar chart for differences with adaptive spacing
    y_pos = np.arange(len(method_labels))
    bar_height, label_font_size = _adjust_bar_spacing_for_methods(method_labels)
    
    # Use difference color if specified, otherwise use default matplotlib colors
    diff_color = plot_colors.get('difference', None)
    bar_kwargs = {'height': bar_height, 'alpha': 0.8, 'edgecolor': 'black', 'linewidth': 0.5}
    if diff_color is not None:
        bar_kwargs['color'] = diff_color
    bars = ax2.barh(y_pos, differences * 100, **bar_kwargs)
    
    # Add value labels with better positioning using vectorized operations
    diffs_array = differences * 100
    widths = np.array([bar.get_width() for bar in bars])
    y_positions = np.array([bar.get_y() + bar.get_height()/2. for bar in bars])
    x_offsets = np.where(widths >= 0, 0.2, -0.2)
    ha_positions = np.where(widths >= 0, 'left', 'right')
    
    # Vectorized label application
    for idx in range(len(bars)):
        ax2.text(widths[idx] + x_offsets[idx], y_positions[idx],
                f'{diffs_array[idx]:+.1f}%', ha=ha_positions[idx], va='center', 
                fontsize=label_font_size)
    
    ax2.set_xlabel('Change in Method Mix (%)', fontsize=8)
    ax2.set_ylabel('Methods', fontsize=8)
    ax2.set_title(chart_titles['method_changes'], fontsize=9, pad=10)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(method_labels, fontsize=label_font_size)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Adjust y-axis limits to give more space
    ax2.set_ylim(-0.5, len(method_labels) - 0.5)
    
    # =============================================================================
    # MIDDLE CENTER: Baseline Pie Chart
    # =============================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create pie chart for baseline
    baseline_pie_cmap = plot_colors.get('baseline_pie', plot_colors.get('pie', 'viridis'))
    baseline_pie_colors = _process_colors(baseline_pie_cmap, len(method_labels))
    wedges, texts, autotexts = ax3.pie(baseline_mix * 100, labels=method_labels, autopct='%1.1f%%', colors=baseline_pie_colors,
                                       startangle=90, textprops={'fontsize': 5})
    ax3.set_title(chart_titles['baseline_pie'], fontsize=9, pad=8)
    
    # =============================================================================
    # MIDDLE RIGHT: Intervention Pie Chart
    # =============================================================================
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Create pie chart for intervention
    intervention_pie_cmap = plot_colors.get('intervention_pie', plot_colors.get('pie', 'viridis'))
    intervention_pie_colors = _process_colors(intervention_pie_cmap, len(method_labels))
    wedges, texts, autotexts = ax4.pie(intervention_mix * 100, labels=method_labels, autopct='%1.1f%%', colors=intervention_pie_colors,
                                       startangle=90, textprops={'fontsize': 5})
    ax4.set_title(chart_titles['intervention_pie'], fontsize=9, pad=8)
    
    # =============================================================================
    # BOTTOM ROW: Heatmap on left, Summary on right
    # =============================================================================
    ax5 = fig.add_subplot(gs[2, 0:2])  # Heatmap takes 2 columns
    
    # Create data for heatmap
    heatmap_data = np.array([baseline_mix * 100, intervention_mix * 100, differences * 100])
    heatmap_labels = ['Baseline (%)', 'Intervention (%)', 'Difference (%)']
    
    # Create heatmap with custom colormap
    heatmap_cmap = plot_colors.get('heatmap', 'viridis')
    im = ax5.imshow(heatmap_data, cmap=heatmap_cmap, aspect='auto', alpha=0.8)
    
    # Add text annotations using vectorized operations
    i_indices, j_indices = np.meshgrid(np.arange(len(heatmap_labels)), 
                                       np.arange(len(method_labels)), 
                                       indexing='ij')
    # Flatten and apply annotations
    for i, j in zip(i_indices.flat, j_indices.flat):
        ax5.text(j, i, f'{heatmap_data[i, j]:.1f}',
                ha="center", va="center", color="black", fontsize=7)
    
    ax5.set_xticks(range(len(method_labels)))
    ax5.set_yticks(range(len(heatmap_labels)))
    ax5.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
    ax5.set_yticklabels(heatmap_labels, fontsize=7)
    ax5.set_title(chart_titles['heatmap'], fontsize=8, pad=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=6)
    
    # =============================================================================
    # Summary statistics on right
    # =============================================================================
    fig.suptitle(title, fontsize=10, y=0.98)
    
    # Add summary text box in a new subplot on the right
    ax6 = fig.add_subplot(gs[2, 2])  # Summary takes 1 column
    ax6.axis('off')  # Hide axes
    
    # Add summary text box
    summary_text = f"""
Summary Statistics:
• Total methods: {len(method_labels)}
• Baseline sum: {np.sum(baseline_mix):.3f}
• Intervention sum: {np.sum(intervention_mix):.3f}
• Max increase: {np.max(differences)*100:+.1f}% ({method_labels[np.argmax(differences)]})
• Max decrease: {np.min(differences)*100:+.1f}% ({method_labels[np.argmin(differences)]})
• Matrices identical: {np.allclose(baseline_mix, intervention_mix)}
"""
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=7,
             verticalalignment='center', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    if save_figure:
        # Save to results directory
        results_dir = _get_results_dir()
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")
    
    if show_figure:
        plt.show()
    
    return fig

# endregion: Plotting functions for METHOD MIX COMPARISON FULL
# region: Plotting functions for PREGNANCIES PER YEAR
def _adjust_bar_spacing_for_methods(method_names, base_height=0.6):
    # Helper function to adjust bar spacing based on number of methods.
    # Returns: tuple (bar_height, y_spacing, font_size) adjusted for the number of methods
    num_methods = len(method_names)
    
    if num_methods <= 5:
        bar_height = 0.7
        font_size = 7
    elif num_methods <= 8:
        bar_height = 0.6
        font_size = 6
    elif num_methods <= 12:
        bar_height = 0.5
        font_size = 6
    else:
        bar_height = 0.4
        font_size = 5
    
    return bar_height, font_size

# endregion: Plotting functions for PREGNANCIES PER YEAR
# region: Plotting functions for METHOD DISTRIBUTION
def plot_pregnancies_per_year(sim1, sim2, show_figure=True, save_figure=False, filename="pregnancies_per_year.png"):
    """
    Plot total new pregnancies per *year* for two FPsim simulations.
    Assumes sim.results.timevec is in years (may include fractional steps).
    """
    def yearly_counts(sim):
        time = sim.results.timevec
        pregnancies = sim.results.fp.pregnancies
        
        # Convert to integer years
        years = np.floor(time).astype(int)
        unique_years = np.unique(years)
        counts = [pregnancies[years == y].sum() for y in unique_years]
        return unique_years, counts

    y1, c1 = yearly_counts(sim1)
    y2, c2 = yearly_counts(sim2)
    color1 = 'blue'
    color2 = 'orange'
    plt.figure(figsize=(8,5))
    plt.plot(y1, c1, marker='o', label=sim1.label)
    plt.fill_between(y1, c1, alpha=0.2, color=color1)
    
    plt.plot(y2, c2, marker='s', label=sim2.label)
    plt.fill_between(y2, c2, alpha=0.2, color=color2)
    plt.xlabel('Year')
    plt.ylabel('New pregnancies per year')
    plt.title('Yearly pregnancy incidence comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_figure:
        # Save to results directory
        results_dir = _get_results_dir()
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")
    
    if show_figure:
        plt.show()
    
    return

# endregion: Plotting functions for METHOD DISTRIBUTION
# region: Plotting functions for METHOD DISTRIBUTION USING BINCCOUNT
def plot_method_bincount(sim1, sim2=None, title="Method Distribution", show_figure=True, 
                          save_figure=False, filename="method_histogram.png", colors=None, exclude_none=True):
    """
    Plot method distribution using bincount approach on sim1.people.fp.method.
    
    This function uses np.bincount to count method usage directly from the method indices
    in the simulation results, which is more efficient than histogram for integer arrays.
    
    Parameters:
    -----------
    sim1 : Sim
        Simulation object (required)
    sim2 : Sim, optional
        Optional second simulation for comparison
    title : str
        Title for the figure
    show_figure : bool
        Whether to display the figure
    save_figure : bool
        Whether to save the figure to file
    filename : str
        Filename for saved figure
    colors : dict or None
        Dictionary with custom colors. Keys can be 'sim1', 'sim2'.
        Values can be colormap names (str), lists of colors, or single colors.
        If None, uses default colors.
    exclude_none : bool, default False
        If True, exclude the 'none' method from the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    
    Examples:
    ---------
    >>> import fpsim as fp
    >>> sim1 = fp.Sim(location='kenya')
    >>> sim1.run()
    >>> plot_method_histogram(sim1)
    
    >>> # Compare two simulations
    >>> sim1 = fp.Sim(location='kenya')
    >>> sim1.run()
    >>> sim2 = fp.Sim(location='kenya', interventions=interventions)
    >>> sim2.run()
    >>> plot_method_histogram(sim1, sim2=sim2, title="Method Comparison")
    """
    # Extract method counts and labels using shared helper (no connector usage)
    if sim2 is not None:
        method_labels, mix1, mix2 = _get_data(
            sim1, sim2, exclude_none=exclude_none
        )
        percentages = mix1 * 100
        percentages2 = mix2 * 100
    else:
        method_labels, mix1 = _get_data(
            sim1, exclude_none=exclude_none
        )
        percentages = mix1 * 100
        percentages2 = None
    
    # Process colors
    if colors is None:
        colors = {}
    sim_color = colors.get('sim1', 'steelblue')
    sim2_color = colors.get('sim2', 'crimson')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(method_labels))
    width = 0.35 if sim2 is not None else 0.6
    
    if sim2 is not None:
        # Side-by-side bars for comparison
        bars1 = ax.bar(x - width/2, percentages, width, label='Simulation 1', 
                       color=sim_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, percentages2, width, label='Simulation 2', 
                       color=sim2_color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels using vectorized operations
        heights1 = np.array([bar.get_height() for bar in bars1])
        heights2 = np.array([bar.get_height() for bar in bars2])
        x_positions1 = np.array([bar.get_x() + bar.get_width()/2. for bar in bars1])
        x_positions2 = np.array([bar.get_x() + bar.get_width()/2. for bar in bars2])
        
        # Vectorized mask for significant values
        mask1 = heights1 > 0.5
        mask2 = heights2 > 0.5
        
        # Apply labels only where significant (vectorized)
        for idx in np.where(mask1)[0]:
            ax.text(x_positions1[idx], heights1[idx] + 0.5, f'{heights1[idx]:.1f}%',
                   ha='center', va='bottom', fontsize=7)
        for idx in np.where(mask2)[0]:
            ax.text(x_positions2[idx], heights2[idx] + 0.5, f'{heights2[idx]:.1f}%',
                   ha='center', va='bottom', fontsize=7)
    else:
        # Single bar chart
        bars = ax.bar(x, percentages, width, color=sim_color, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Add value labels using vectorized operations
        heights = np.array([bar.get_height() for bar in bars])
        x_positions = np.array([bar.get_x() + bar.get_width()/2. for bar in bars])
        
        # Vectorized mask for significant values
        mask = heights > 0.5
        
        # Apply labels only where significant (vectorized)
        for idx in np.where(mask)[0]:
            ax.text(x_positions[idx], heights[idx] + 0.5, f'{heights[idx]:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Contraceptive Methods', fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=8)
    
    # if sim2 is not None:
    #     ax.legend(fontsize=8)
    ax.legend(fontsize=8, loc='best')   
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_figure:
        # Save to results directory
        results_dir = _get_results_dir()
        save_path = results_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved as: {save_path}")
    
    if show_figure:
        plt.show()
    
    return fig

# endregion: Plotting functions for METHOD DISTRIBUTION USING BINCCOUNT
# region: Plotting functions for METHOD MIX BY TYPE PER YEAR
def plot_method_mix_by_type_per_year(sim, interventions=None):
    method_mix = sim.connectors.fp.method_mix  # Shape: (n_options, n_timepoints)
    _, n_timepoints = method_mix.shape
    years = sim.results.timevec.years
    if len(years) != n_timepoints:
        years = years[:n_timepoints]
    methods_dict = sim.connectors.contraception.methods
    name_to_type = {
        'pill': 'pill',
        'iud': 'iud',
        'inj': 'injectable',
        'impl': 'impl',
        'cond': 'cond',
        'btl': 'btl',
        'wdraw': 'wdraw',
        'othtrad': 'othtrad',
        'othmod': 'othmod',
        'none': 'none'
    }
    
    # Get method type for each method
    method_types = {}
    for _, method in methods_dict.items():
        if hasattr(method, 'method_type'):
            method_type = method.method_type
        else:
            method_type = name_to_type.get(method.name, 'other_modern')
        method_types[method.idx] = method_type
    
    # Group methods by type and sum proportions
    type_to_indices = {}
    for method_idx, method_type in method_types.items():
        if method_type not in type_to_indices:
            type_to_indices[method_type] = []
        type_to_indices[method_type].append(method_idx)
    
    # Calculate sum per type per year
    type_mix = {}
    for method_type, indices in type_to_indices.items():
        if method_type == 'none':
            continue  # Skip 'none' method
        # Sum proportions for all methods of this type across all timepoints
        # method_mix[indices, :] selects rows for methods of this type, all columns (timepoints)
        type_mix[method_type] = method_mix[indices, :].sum(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get colors for each type
    colors = plt.cm.Set3(np.linspace(0, 1, len(type_mix)))
    
    for i, (method_type, proportions) in enumerate(type_mix.items()):
        ax.plot(years, proportions * 100, label=method_type, linewidth=2, color=colors[i])

    fig = _add_annotations(fig, interventions)    
    # # Add annotations for methods added via copy_from_existing
    # if interventions is not None:
    #     intervention_events = []
    #     for intv in interventions:
    #         # Check if this is a copy_from_existing intervention
    #         if hasattr(intv, 'name') and 'copy_from_existing' in str(intv.name):
    #             if hasattr(intv, 'pars') and hasattr(intv.pars, 'new_method') and intv.pars.new_method is not None:
    #                 method = intv.pars.new_method
    #                 year = intv.pars.year
    #                 intervention_events.append({
    #                     'year': year,
    #                     'method_label': method.label,
    #                     'method_name': method.name
    #                 })
        
    #     # Add vertical lines and annotations for each copy_from_existing event
    #     if intervention_events:
    #         # Get current y-limits after plotting
    #         y_min, y_max = ax.get_ylim()
    #         y_range = y_max - y_min
            
    #         for event_index, event in enumerate(intervention_events):
    #             year = event['year']
    #             method_label = event['method_label']
                
    #             # Add vertical line
    #             ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                
    #             # Add text annotation
    #             # Position annotation at different heights to avoid overlap (from top of plot)
    #             annotation_offset = (event_index % 3 + 1) * (y_range * 0.08)
    #             y_pos = y_max - annotation_offset
                
    #             ax.annotate(
    #                 f'+ {method_label}',
    #                 xy=(year, y_pos),
    #                 xytext=(5, 0),
    #                 textcoords='offset points',
    #                 fontsize=9,
    #                 alpha=0.7,
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5),
    #                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.5, lw=0.5)
    #             )
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion (%)', fontsize=12)
    ax.set_title('Method Mix by Method Type Over Time', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(years[0], years[-1])
    
    plt.tight_layout()
    # Save to results directory
    results_dir = _get_results_dir()
    save_path = results_dir / 'method_mix_by_type_per_year.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved as '{save_path}'")
    plt.show()
    
    return fig

# endregion: Plotting functions for METHOD MIX BY TYPE PER YEAR
# region: Plotting functions for ITERABLE PLOTS
def iterable_plots(sim, interventions=None):
        
    fig = go.Figure()
    # Get methods sorted by idx to match method_mix indices
    # method_mix[i] corresponds to method.idx == i
    methods_sorted = sorted(sim.connectors.contraception.methods.values(), key=lambda m: m.idx)
    names = [m.name for m in methods_sorted]

    # Plot method mix lines
    # method_mix[i] index corresponds to method.idx == i
    for i in range(0, 10):
        fig.add_trace(go.Scatter(
            x=sim.results.timevec.years,
            y=sim.connectors.fp.method_mix[i],
            mode='lines',
            name=names[i],
            line=dict(width=2)
        ))

    # Add annotations for methods added via copy_from_existing
    if interventions is not None:
        intervention_events = []
        for intv in interventions:
            # Check if this is a copy_from_existing intervention
            if hasattr(intv, 'name') and 'copy_from_existing' in str(intv.name):
                if hasattr(intv, 'pars') and hasattr(intv.pars, 'new_method') and intv.pars.new_method is not None:
                    method = intv.pars.new_method
                    year = intv.pars.year
                    intervention_events.append({
                        'year': year,
                        'method_label': method.label,
                        'method_name': method.name
                    })
        
        # Add vertical lines and annotations for each copy_from_existing event
        if intervention_events:
            # Get y-range from the data
            all_y_values = []
            for i in range(0, 10):
                all_y_values.extend(sim.connectors.fp.method_mix[i])
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max - y_min
            
            for event_index, event in enumerate(intervention_events):
                year = event['year']
                method_label = event['method_label']
                
                # Add vertical line
                fig.add_vline(
                    x=year,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    line_width=1
                )
                
                # Add text annotation
                # Position annotation at different heights to avoid overlap (from top of plot)
                annotation_offset = (event_index % 3 + 1) * (y_range * 0.08)
                y_pos = y_max - annotation_offset
                
                fig.add_annotation(
                    x=year,y=y_pos,
                    text=f'+ {method_label}',
                    showarrow=True,
                    arrowhead=1, arrowwidth=1, arrowcolor="gray",
                    ax=10, ay=0,
                    bgcolor="lightblue",
                    bordercolor="lightblue",
                    borderwidth=1,
                    font=dict(size=9, color="black"),
                    opacity=0.7
                )

    # Update layout
    fig.update_layout(
        title='Method Mix by Method Type Over Time',
        xaxis_title='Year',
        yaxis_title='Method Mix',
        width=1200,
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    fig.show()

    return fig

# endregion: Plotting functions for ITERABLE PLOTS
#region: Plotting functions for NEW METHODs

COLORS = {
    'baseline': '#2E86AB',
    'intervention': '#A23B72',
    'new_method': '#F18F01',
}

def plot_injectable_methods_comparison(
    baseline_sim,
    intervention_sim,
    start_year,
    end_year,
    location,
    intervention_year=None,
    save_path='add_method_injectables.png',
):
    """
    Plot injectable method usage comparison over time.

    Args:
        baseline_sim: Simulation without the intervention.
        intervention_sim: Simulation with the new method intervention.
        start_year / end_year: Year range for the plot.
        location (str): Name used for titles/labels.
        intervention_year (float, optional): Year to mark with a vertical “program start” line.
        save_path (Path-like): Where to write the figure.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get years as numeric array
    years = np.linspace(start_year, end_year, len(baseline_sim.results.timevec))
    
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
    
    if intervention_year is not None:
        ax1.axvline(
            intervention_year,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label='Program Start',
        )
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
    if intervention_year is not None:
        ax2.axvline(
            intervention_year,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
        )
    
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
    
    fig.suptitle(f'Injectable Methods Comparison in {location.title()}', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved injectable comparison to {save_path}")
    return fig


def plot_method_mix_evolution(sim, start_year, end_year, intervention_year, location, title='Method Mix Over Time', save_path='add_method_mix.png'):
    """Plot stacked area chart of method mix over time."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Get method mix data from FPmod
    fp_mod = sim.connectors['fp']
    method_mix = fp_mod.method_mix
    # Get years as numeric array
    years = np.linspace(start_year, end_year, len(sim.results.timevec))
    
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
    
    ax.axvline(intervention_year, color='red', linestyle='--', linewidth=2, 
               label='MY-NEW-METHOD Introduced', alpha=0.7)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Proportion of Users', fontsize=12)
    ax.set_title(f'{title} in {location.title()}', fontsize=14, fontweight='bold')
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


def plot_new_method_adoption(sim, start_year, end_year, intervention_year, location, save_path='add_method_adoption.png'):
    """Plot adoption of the new method over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get method mix data
    fp_mod = sim.connectors['fp']
    method_mix = fp_mod.method_mix
    # Get years as numeric array
    years = np.linspace(start_year, end_year, len(sim.results.timevec))
    
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
        ax.axvline(intervention_year, color='red', linestyle='--', linewidth=2, 
                  label='Introduction Year', alpha=0.7)
        
        # Add annotations
        final_adoption = my_new_method_proportion[-1] * 100
        if final_adoption > 0.1:
            ax.annotate(f'Final adoption:\n{final_adoption:.2f}%', 
                       xy=(end_year, my_new_method_proportion[-1] * 100),
                       xytext=(end_year - 3, final_adoption + 2),
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
    ax.set_title(f'MY-NEW-METHOD Adoption in {location.title()} - Comprehensive Program Impact', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([start_year, end_year])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved new method adoption to {save_path}")
    return fig


def plot_method_comparison_bar(baseline_sim, intervention_sim, start_year, end_year, intervention_year, location, save_path='add_method_bar.png'):
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
    ax.set_title(f'Contraceptive Method Usage at End of Simulation ({end_year})', 
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


def plot_births_comparison(baseline_sim, intervention_sim, start_year, end_year, intervention_year, location, save_path='add_method_births.png'):
    """Plot births over time comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get years as numeric array
    years = np.linspace(start_year, end_year, len(baseline_sim.results.timevec))
    
    # Get births
    baseline_births = baseline_sim.results.fp.births
    intervention_births = intervention_sim.results.fp.births
    
    # Plot
    ax.plot(years, baseline_births, label='Baseline', 
           color=COLORS['baseline'], linewidth=2, marker='o', markersize=3)
    ax.plot(years, intervention_births, label='With MY-NEW-METHOD', 
           color=COLORS['intervention'], linewidth=2, marker='s', markersize=3)
    ax.axvline(intervention_year, color='gray', linestyle='--', alpha=0.5, label='Intervention')
    
    # Calculate and show cumulative difference
    years_mask = years >= intervention_year
    baseline_total = np.sum(baseline_births[years_mask])
    intervention_total = np.sum(intervention_births[years_mask])
    births_averted = baseline_total - intervention_total
    
    ax.text(0.02, 0.98, f'Births after intervention:\nBaseline: {int(baseline_total)}\nWith MY-NEW-METHOD: {int(intervention_total)}\nDifference: {int(births_averted)}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Births per Month', fontsize=12)
    ax.set_title(f'Births Over Time: Baseline vs. MY-NEW-METHOD Program in {location.title()}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved births comparison to {save_path}")
    return fig


def create_summary_figure(baseline_sim, intervention_sim, start_year, end_year, intervention_year, location, save_path='add_method_summary.png'):
    """Create a comprehensive summary figure with multiple subplots."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # Get years as numeric array
    years = np.linspace(start_year, end_year, len(baseline_sim.results.timevec))
    
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
    ax1.axvline(intervention_year, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Introduction')
    
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
    ax2.axvline(intervention_year, color='red', linestyle='--', alpha=0.5)
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
    ax3.axvline(intervention_year, color='red', linestyle='--', alpha=0.5)
    
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
    ax5.set_title(f'Top 6 Methods by Usage ({end_year})', fontweight='bold', fontsize=12)
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
    ax6.set_title(f'All Methods: Final Usage Comparison ({end_year})', fontweight='bold', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=9)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(f'Impact of Introducing MY-NEW-METHOD in {location.title()} ({start_year}-{end_year}) - Comprehensive Program', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary figure to {save_path}")
    return fig


def print_summary_statistics(baseline_sim, intervention_sim, start_year, end_year, intervention_year):
    """Print summary statistics comparing both simulations."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Final CPR/mCPR
    print(f"\nFinal Prevalence Rates ({end_year}):")
    print(f"{'Metric':<20} {'Baseline':>15} {'With MY-NEW-METHOD':>15} {'Change':>15}")
    print("-" * 70)
    
    baseline_mcpr = baseline_sim.results.contraception.mcpr[-1]
    interv_mcpr = intervention_sim.results.contraception.mcpr[-1]
    print(f"{'mCPR':<20} {baseline_mcpr:>14.3f} {interv_mcpr:>14.3f} {interv_mcpr-baseline_mcpr:>+14.3f}")
    
    baseline_cpr = baseline_sim.results.contraception.cpr[-1]
    interv_cpr = intervention_sim.results.contraception.cpr[-1]
    print(f"{'CPR':<20} {baseline_cpr:>14.3f} {interv_cpr:>14.3f} {interv_cpr-baseline_cpr:>+14.3f}")
    
    # Total births
    years_numeric = np.linspace(start_year, end_year, len(baseline_sim.results.timevec))
    years_mask = years_numeric >= intervention_year
    baseline_births = np.sum(baseline_sim.results.fp.births[years_mask])
    interv_births = np.sum(intervention_sim.results.fp.births[years_mask])
    births_averted = baseline_births - interv_births
    
    print(f"\nBirths After Intervention ({intervention_year}-{end_year}):")
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

# endregion: PLOTTING FUNCTIONS FOR NEW METHOD