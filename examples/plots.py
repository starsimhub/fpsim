import numpy as np
import matplotlib.pyplot as plt
from fpsim.sim import Sim
from fpsim.interventions import update_methods
# from fpsim.intervention_builder import MethodType
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
           'plot_method_mix_by_type_per_year', 'iterable_plots', 'plot_pregnancies_per_year']


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
    results_dir = script_dir / 'results'
    
    # Create the directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)
    
    return results_dir


class MethoType(Enum):
    PILL = 'Pill'
    IUD = 'IUD'
    INJECTABLE = 'Injectable'
    CONDOM = 'Condom'
    BTL = 'BTL'
    WITHDRAWAL = 'Withdrawal'
    IMPLANT = 'Implant'
    OTHER_TRADITIONAL = 'Other traditional'
    OTHER_MODERN = 'Other modern'
    NONE = 'None'

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
    # Uses np.bincount for efficient counting without connector dependencies.
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
    
    # Extract method data from all simulations
    method_data_arrays = [sim.people.fp.method for sim in sims]
    
    # Calculate max method index across all simulations
    max_indices = [arr.max() + 1 if len(arr) > 0 else 1 for arr in method_data_arrays]
    max_method_idx = max(max_indices) if max_indices else 1
    
    # Count methods using bincount for each simulation
    counts = [np.bincount(arr, minlength=max_method_idx) for arr in method_data_arrays]
    
    # Normalize to proportions (0-1 range)
    method_mix_arrays = []
    for count in counts:
        total = np.sum(count)
        mix = (count / total) if total > 0 else count
        method_mix_arrays.append(np.array(mix))
    
    # Create method labels using standard mapping (already in index order: 0, 1, 2, ...)
    method_labels = [METHOD_INDEX_TO_NAME.get(i, f'Method_{i}') for i in range(max_method_idx)]
    method_indices = np.arange(max_method_idx)
    if exclude_none:
        # Exclude the 'none' method (index 0) and renormalize proportions
        none_idx = 0
        mask = method_indices != none_idx
        method_indices = method_indices[mask]
        method_labels = [method_labels[i] for i in method_indices]
        
        # Filter and renormalize all method mix arrays (maintains index order)
        filtered_arrays = []
        for mix_array in method_mix_arrays:
            filtered = mix_array[mask]
            total = np.sum(filtered)
            if total > 0:
                filtered = filtered / total
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