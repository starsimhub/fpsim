"""
User-friendly interface for modeling contraceptive method interventions.

This module provides an easy-to-use interface for program teams to model
contraceptive interventions without needing to understand the complex internal FPsim structure.

What you can model:
-------------------
1. **Efficacy changes**: Model improved method effectiveness (e.g., better quality pills)
2. **Duration changes**: Model how long people stay on a method (e.g., better counseling, 
   reduced side effects, improved supply chain)
3. **Method mix**: Model shifts in which methods people choose (e.g., social marketing campaigns)
4. **Probability of use**: Model changes in overall contraceptive uptake
5. **Switching patterns**: Model how people transition between methods

Key concepts :
---------------------------
- **Efficacy**: How well the method prevents pregnancy (0 to 1, where 0.99 = 99% effective)
- **Duration**: How many months people typically stay on the method before discontinuing
- **Method mix**: The distribution of users across different contraceptive methods
- **Switching matrix**: The probabilities of moving from one method to another

Note: Small changes in duration can have large effects on method mix because people
accumulate on methods they stay on longer. This is realistic - improving continuation
rates is one of the most impactful interventions.

Quick Reference 
------------------------------
**What intervention should I model for...**

1. **Better counseling / side effect management** → `set_duration_months()`
   People stay on methods longer with better support
   
2. **Improved product quality** → `set_efficacy()`
   Better manufacturing or user education increases effectiveness
   
3. **Social marketing campaign for specific method** → `set_method_mix()`
   Shift demand toward targeted methods (requires baseline capture)
   
4. **General awareness campaign** → `set_probability_of_use()`
   Increase overall contraceptive use (CPR)
   
5. **Provider training on method transitions** → `scale_switching_matrix()`
   Make it easier for people to switch to better methods
   
6. **Comprehensive program** → Combine multiple methods!
   Example: Better counseling (duration) + quality improvements (efficacy)

**Method name shortcuts:**
- 'pill' = Pills
- 'inj' = Injectables (DMPA, Sayana Press, etc.)
- 'impl' = Implants (Jadelle, Implanon, etc.)
- 'iud' = IUDs
- 'cond' = Male condoms
- 'btl' = Female sterilization (tubal ligation)
- 'wdraw' = Withdrawal
- 'othtrad' = Other traditional methods
- 'othmod' = Other modern methods
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, singledispatch
from operator import attrgetter
from pathlib import Path
from typing import Dict, Optional, Sequence, Union, TypedDict, List, Any

import numpy as np
import sciris as sc

from . import interventions as fpi
from .methods import NewMethodConfig

__all__ = [
    'MethodIntervention',
    'make_update_methods',
    'MethodName',
]


# ==========================================
# Enums and Type Definitions
# ==========================================

class MethodName(str, Enum):
    """Enumeration of standard contraceptive method names."""
    NONE = 'none'
    PILL = 'pill'
    IUD = 'iud'
    INJECTABLE = 'inj'
    CONDOM = 'cond'
    BTL = 'btl'
    WITHDRAWAL = 'wdraw'
    IMPLANT = 'impl'
    OTHER_TRAD = 'othtrad'
    OTHER_MOD = 'othmod'
    
    @classmethod
    def values(cls) -> List[str]:
        """Return list of all method name values."""
        return [m.value for m in cls]
    
    @classmethod
    def excluding_none(cls) -> List[str]:
        """Return list of method names excluding 'none'."""
        return [m.value for m in cls if m != cls.NONE]


class PreviewDict(TypedDict, total=False):
    """Typed dictionary for preview() return value."""
    year: float
    efficacy: Optional[Dict[str, float]]
    duration_months: Optional[Dict[str, Union[int, float]]]
    p_use: Optional[float]
    method_mix: Optional[Dict[str, float]]
    switching_matrix: Optional[str]
    new_method: Optional[Dict[str, Any]]
    label: Optional[str]


@dataclass(slots=True)
class InterventionConfig:
    """Container for intervention parameters with default values."""
    eff: Dict[str, float] = field(default_factory=dict)
    dur: Dict[str, Union[int, float]] = field(default_factory=dict)
    p_use: Optional[float] = None
    mix_values: Dict[str, float] = field(default_factory=dict)
    method_mix_base: Optional[np.ndarray] = None
    switch: Optional[dict] = None
    new_method: Optional[NewMethodConfig] = None


# ==========================================
# Utility Functions
# ==========================================

@singledispatch
def load_config(source) -> dict:
    """Load configuration from various sources."""
    raise ValueError(f"Unsupported type: {type(source)}. Expected str, Path, or dict.")


@load_config.register(str)
def _(path: str) -> dict:
    """Load from string path."""
    return load_config(Path(path))


@load_config.register(Path)
def _(path: Path) -> dict:
    """Load from Path object."""
    if path.suffix == '.json':
        return json.loads(path.read_text())
    elif path.suffix in ('.yml', '.yaml'):
        try:
            import yaml
        except ImportError as e:
            raise ImportError('YAML support not available. Please install pyyaml to load YAML files.') from e
        return yaml.safe_load(path.read_text())
    raise ValueError(f'Unsupported file type: {path.suffix}. Expected .json, .yml, or .yaml')


@load_config.register(dict)
def _(data: dict) -> dict:
    """Return copy of dictionary."""
    return sc.dcp(data)


@contextmanager
def _print_if(condition: bool, message: str):
    """Context manager for conditional printing."""
    try:
        yield
        if condition:
            print(message)
    except Exception:
        raise


def validate_method_name(allow_new: bool = True):
    """
    Decorator to validate and normalize method names.
    
    Args:
        allow_new: If True, allows method names not in the standard list
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, method: str, *args, **kwargs):
            normalized = self._normalize_name(method, allow_new=allow_new)
            return func(self, normalized, *args, **kwargs)
        return wrapper
    return decorator


# ==========================================
# Main Class
# ==========================================

class MethodIntervention:
    """
    Builder for creating contraceptive method interventions with a user-friendly API.

    This class helps you model "what-if" scenarios for contraceptive programs:
    - What if we improve injectable continuation rates?
    - What if we run a campaign to increase pill efficacy (quality)?
    - What if we shift more people toward LARCs?
    
    Usage Pattern
    -------------
    1. Create a modifier: `mod = fp.MethodIntervention(year=2025)`
    2. Configure changes: `mod.set_efficacy('pill', 0.95).set_duration_months('pill', 18)`
    3. Build intervention: `intv = mod.build()`
    4. Run simulation: `sim = fp.Sim(pars=pars, interventions=intv)`
    
    Real-world Examples
    -------------------
    **Improving method quality:**
    >>> mod = MethodIntervention(year=2025)
    >>> mod.set_efficacy('pill', 0.95)  # Improve pill effectiveness to 95%
    
    **Better counseling increases continuation:**
    >>> mod = MethodIntervention(year=2025)
    >>> mod.set_duration_months('inj', 36)  # Set target duration to 36 months
    
    **Social marketing campaign shifts method mix:**
    >>> mod = MethodIntervention(year=2025)
    >>> mod.capture_method_mix_from_sim(baseline_sim)
    >>> mod.set_method_mix('impl', 0.25)  # Target 25% of users choosing implants
    
    Important Notes 
    ------------------------------
    - **Duration impacts**: Duration changes translate directly to usage changes in steady state.
      Increasing duration by 50% increases method usage by 50% because people accumulate on
      the method over time. This is realistic and expected!
    - **Method competition**: When one method improves, similar methods often decrease
      (e.g., improving injectables may decrease implants as they compete for users)
    - **Units**: Efficacy is 0-1 (e.g., 0.95 = 95%), Duration is in months (e.g., 24 = 2 years)
    - **Mix values**: Can provide as percentages (25) or fractions (0.25), both work
    """
    
    __slots__ = ('year', 'label', '_config')

    # Standard method names and orderings
    _method_names = MethodName.values()
    _method_mix_order = MethodName.excluding_none()
    
    # Mapping from method.name to method.label (for interventions)
    _name_to_label = {
        'none': 'None',
        'pill': 'Pill',
        'iud': 'IUDs',
        'inj': 'Injectables',
        'cond': 'Condoms',
        'btl': 'BTL',
        'wdraw': 'Withdrawal',
        'impl': 'Implants',
        'othtrad': 'Other traditional',
        'othmod': 'Other modern',
    }

    def __init__(self, year: float, label: Optional[str] = None):
        self.year: float = year
        self.label: Optional[str] = label
        self._config = InterventionConfig()

    # -----------------
    # Properties
    # -----------------
    @property
    def method_mix_order(self) -> List[str]:
        """List of contraceptive methods available for mix calculations (excludes 'none')."""
        return self._method_mix_order
    
    @property
    def has_efficacy_changes(self) -> bool:
        """Check if any efficacy changes are configured."""
        return bool(self._config.eff)
    
    @property
    def has_duration_changes(self) -> bool:
        """Check if any duration changes are configured."""
        return bool(self._config.dur)
    
    @property
    def has_method_mix_changes(self) -> bool:
        """Check if any method mix changes are configured."""
        return bool(self._config.mix_values)
    
    @property
    def has_switching_matrix_changes(self) -> bool:
        """Check if switching matrix is modified."""
        return self._config.switch is not None
    
    @property
    def has_new_method(self) -> bool:
        """Check if a new method is being added."""
        return self._config.new_method is not None
    
    # -----------------
    # Helper utilities
    # -----------------
    def _normalize_name(self, name: str, allow_new: bool = True) -> str:
        """
        Validate method name.
        
        Args:
            name: Method name to validate
            allow_new: If True, allows method names not in the standard list
                      (for dynamically added methods)
        """
        if not isinstance(name, str):
            raise ValueError('Method name must be a string')
        if name not in self._method_names:
            if not allow_new:
                raise ValueError(f'Method name must be one of: {self._method_names}')
            # Method might be dynamically added, allow it but don't convert label
        return name

    @staticmethod
    def _load_file(path: str) -> dict:
        """
        Load configuration file (deprecated - use load_config instead).
        
        This method is kept for backward compatibility.
        """
        return load_config(path)

    def _convert_mix_value(self, value: float) -> float:
        val = float(value)
        if val < 0:
            raise ValueError('Method mix entries must be non-negative')
        if val > 1.0 + 1e-6:
            val = val / 100.0
        return val

    def _ensure_method_mix_base(self) -> np.ndarray:
        if self._config.method_mix_base is None:
            raise ValueError(
                'Method mix baseline not set. Call `set_method_mix_baseline()` or '
                '`capture_method_mix_from_sim(sim)` before modifying the mix.'
            )
        return self._config.method_mix_base

    def _build_method_mix_array(self) -> Optional[np.ndarray]:
        """Calculate normalized method mix array from baseline and targeted method changes."""
        if not self._config.mix_values:
            return None

        base = self._ensure_method_mix_base()
        arr = base.copy()

        # Track which methods have been explicitly targeted
        n_methods = len(arr)
        targeted_mask = np.zeros(n_methods, dtype=bool)
        
        # Apply targeted method mix values
        for name, val in self._config.mix_values.items():
            if name == 'none':
                raise ValueError('Method mix cannot be set for "none"; use probability of use instead.')
            if name not in self.method_mix_order:
                raise ValueError(f'Method mix updates must reference one of: {self.method_mix_order}')
            idx = self.method_mix_order.index(name)
            new_val = self._convert_mix_value(val)
            arr[idx] = new_val
            targeted_mask[idx] = True

        # Calculate total of targeted methods
        targeted_sum = arr[targeted_mask].sum()
        
        if targeted_sum > 1.0 + 1e-6:
            raise ValueError('Method mix updates sum to more than 1.0; reduce the requested values.')

        # Calculate remaining probability for non-targeted methods
        untargeted_mask = ~targeted_mask
        remaining_total_base = base[untargeted_mask].sum()

        if remaining_total_base < 1e-12:
            if targeted_sum < 1.0 - 1e-6:
                raise ValueError('Baseline for non-targeted methods sums to zero; cannot redistribute remaining probability.')
            scale = 0.0
        else:
            scale = max(0.0, (1.0 - targeted_sum) / remaining_total_base)

        # Scale non-targeted methods proportionally
        arr[untargeted_mask] = base[untargeted_mask] * scale

        # Normalize to ensure sum equals 1.0
        total = arr.sum()
        if total <= 0:
            raise ValueError('Provided method mix sums to zero after applying updates')
        return arr / total

    # -----------------
    # Builder methods
    # -----------------
    @validate_method_name(allow_new=True)
    def set_efficacy(self, method: str, efficacy: float, print_efficacy=False) -> 'MethodIntervention':
        """
        Set the contraceptive efficacy (failure rate prevention) for a method.
        
        **What this models:** Changes in how well a method prevents pregnancy, which could result from:
        - Improved product quality (e.g., better manufacturing standards)
        - Better user education (e.g., teaching correct condom use)
        - Improved formulations (e.g., higher-dose pills)
        
        Parameters
        ----------
        method : str
            Method name: 'pill', 'iud', 'inj' (injectables), 'impl' (implants), 
            'cond' (condoms), 'btl' (tubal ligation), 'wdraw' (withdrawal), etc.
        efficacy : float
            Effectiveness as a decimal between 0 and 1
            Examples: 0.95 = 95% effective, 0.99 = 99% effective
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining (so you can do multiple .set_X() calls)
            
        Examples
        --------
        >>> # Model improved pill quality
        >>> mod = MethodIntervention(year=2025)
        >>> mod.set_efficacy('pill', 0.95)  # Increase from baseline to 95%
        
        >>> # Model better condom education
        >>> mod.set_efficacy('cond', 0.96)  # Increase correct use effectiveness
        
        Notes 
        --------------------
        - Typical efficacies: Pills ~94%, IUDs ~99%, Implants ~99%, Condoms ~95%
        - Small changes (2-3%) can have meaningful impact on unintended pregnancies
        - Efficacy changes affect pregnancy outcomes but don't directly change method mix
        """
        val = float(efficacy)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f'Efficacy for {method} must be between 0 and 1')
        
        with _print_if(print_efficacy, f'Efficacy: {method} = {val}'):
            self._config.eff[method] = val
        
        return self

    @validate_method_name(allow_new=True)
    def set_duration_months(self, method: str, months: Union[int, float], print_duration=False) -> 'MethodIntervention':
        """
        Set how long (in months) people typically stay on a contraceptive method.
        
        **What this models:** Continuation rates - how long people keep using a method before 
        stopping. This is one of the MOST IMPACTFUL interventions you can model. Improvements 
        could result from:
        - Better counseling and side effect management
        - Improved supply chain (reducing stockouts)
        - Reduced side effects (better formulations)
        - Better follow-up and support services
        - Community education reducing stigma
        
        Parameters
        ----------
        method : str
            Method name: 'pill', 'iud', 'inj', 'impl', 'cond', 'btl', 'wdraw', etc.
        months : int or float
            Average duration in months that people stay on the method
            Examples: 12 = 1 year, 24 = 2 years, 36 = 3 years
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Model improved counseling that increases injectable continuation
        >>> mod = MethodIntervention(year=2025)
        >>> mod.set_duration_months('inj', 30)  # Set target duration to 30 months
        
        >>> # Model supply chain improvements reducing pill stockouts
        >>> mod.set_duration_months('pill', 15)  # Set target duration to 15 months
        
        >>> # Model comprehensive program improving multiple methods
        >>> mod.set_duration_months('inj', 30).set_duration_months('impl', 40)
        
        Critical Notes 
        -----------------------------
        **Duration changes have LARGE effects on method mix:**
        - Duration changes translate DIRECTLY to usage changes in steady state
        - This is because people "accumulate" on methods they stay on longer
        - Example: If 100 people/month start injectables:
          * At 24 months average duration → ~2,400 current users at any time
          * At 36 months average duration → ~3,600 current users at any time
          * That's a 50% duration increase → 50% usage increase!
        
        **Why this is realistic:**
        - Duration/continuation is one of the most impactful real-world program interventions
        - Studies show counseling can increase continuation by 30-60%
        - The large method mix changes you see in results are expected, not bugs!
        
        **Note on baseline durations:**
        Baseline values are calibrated to country-specific data and vary by location.
        Use `capture_method_mix_from_sim()` or check your location's calibration parameters
        to understand current duration distributions before setting intervention targets.
        """
        mval = float(months)
        if mval <= 0:
            raise ValueError(f'Duration (months) for {method} must be positive')
        
        with _print_if(print_duration, f'Duration: {method} = {mval} months'):
            self._config.dur[method] = mval
        
        return self

    def set_probability_of_use(self, p_use: float, print_p_use=False) -> 'MethodIntervention':
        """
        Set the overall probability that eligible people use ANY contraceptive method.
        
        **What this models:** Changes in overall contraceptive uptake/coverage (CPR), which 
        could result from:
        - Mass media campaigns increasing awareness
        - Reducing access barriers (cost, distance, stigma)
        - Community health worker outreach programs
        - Policy changes (e.g., free contraception)
        - Male engagement programs
        
        Parameters
        ----------
        p_use : float
            Probability between 0 and 1 that eligible people use contraception
            Examples: 0.40 = 40% CPR, 0.58 = 58% CPR
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Model mass media campaign increasing uptake from 40% to 50%
        >>> mod = MethodIntervention(year=2025)
        >>> mod.set_probability_of_use(0.50)
        
        >>> # Combine with method mix to model targeted LARC campaign
        >>> mod.set_probability_of_use(0.55)  # Increase overall use
        >>> mod.set_method_mix('impl', 0.20)  # Target 20% choosing implants
        
        Notes 
        --------------------
        - This affects overall contraceptive prevalence (CPR)
        - Use this for interventions that increase access/demand broadly
        - For method-specific interventions, use set_duration_months or set_method_mix instead
        - Note: May not work with all FPsim choice modules (the model will warn you)
        """
        p = float(p_use)
        if not (0.0 <= p <= 1.0):
            raise ValueError('Probability of use must be between 0 and 1')
        
        with _print_if(print_p_use, f'Probability of use: {p}'):
            self._config.p_use = p
        
        return self

    @validate_method_name(allow_new=True)
    def set_method_mix(self, method: str, value: float, baseline_sim=None, print_method_mix=False) -> 'MethodIntervention':
        """
        Set the target share/percentage for a specific contraceptive method.
        
        **What this models:** Changes in which methods people choose (method mix), which 
        could result from:
        - Social marketing campaigns promoting specific methods
        - Provider training on method-specific counseling
        - Introduction of new methods or brands
        - Changes in method availability at service delivery points
        - Community-based distribution focusing on specific methods
        
        Parameters
        ----------
        method : str
            Method name: 'pill', 'iud', 'inj', 'impl', 'cond', etc.
        value : float
            Target share as decimal (0.25 = 25%) or percentage (25 = 25%)
            The tool will auto-detect and normalize either format
        baseline_sim : fp.Sim, optional
            Baseline simulation to capture current method mix from.
            Only needed if you haven't already called capture_method_mix_from_sim().
            The simulation must be initialized (sim.init() must have been called).
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Simple usage with baseline_sim parameter (recommended)
        >>> baseline_sim = fp.Sim(pars=pars)
        >>> baseline_sim.init()
        >>> mod = MethodIntervention(year=2025)
        >>> mod.set_method_mix('impl', 0.30, baseline_sim=baseline_sim)  # Auto-captures baseline
        
        >>> # Multi-method targeting (baseline captured on first call)
        >>> mod.set_method_mix('inj', 0.25, baseline_sim=baseline_sim)
        >>> mod.set_method_mix('impl', 0.20)  # Baseline already captured
        
        >>> # Alternative: capture once, then set multiple
        >>> mod.capture_method_mix_from_sim(baseline_sim)
        >>> mod.set_method_mix('inj', 0.25)
        >>> mod.set_method_mix('impl', 0.20)
        
        How Method Mix Adjustment Works
        --------------------------------
        1. You set target shares for one or more methods (e.g., implants = 30%)
        2. Other methods keep their relative proportions but scale down to make room
        3. Everything sums to 100% automatically
        
        Example: If baseline is [Pills: 20%, Injectables: 40%, Implants: 40%]
        And you set implants to 50%, result will be:
        [Pills: 10%, Injectables: 20%, Implants: 50%]
        (Pills and injectables scaled down proportionally)
        
        Notes 
        --------------------
        - Pass baseline_sim parameter for convenience, or capture explicitly with capture_method_mix_from_sim()
        - You can target multiple methods in one intervention
        - Non-targeted methods adjust automatically (proportional rescaling)
        - Use percentages (25) or decimals (0.25), both work
        - Method mix changes model demand-side shifts, not supply constraints
        """
        # Auto-capture baseline if provided and not already set
        if baseline_sim is not None and self._config.method_mix_base is None:
            self.capture_method_mix_from_sim(baseline_sim)
        
        val = float(value)
        if val < 0:
            raise ValueError(f'Method mix for {method} cannot be negative')
        
        with _print_if(print_method_mix, f'Method mix: {method} = {val}'):
            self._config.mix_values[method] = val
        
        return self

    def set_method_mix_baseline(self, mix_values: Sequence[float]) -> 'MethodIntervention':
        """Store the baseline method mix so that only requested entries are updated."""
        arr = np.asarray(mix_values, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Baseline method mix must be a one-dimensional sequence')
        if arr.size != len(self.method_mix_order):
            raise ValueError(f'Baseline method mix must have {len(self.method_mix_order)} entries')
        if (arr < 0).any():
            raise ValueError('Baseline method mix cannot contain negative values')
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 100.0
        total = arr.sum()
        if total <= 0:
            raise ValueError('Baseline method mix sums to zero')
        self._config.method_mix_base = arr / total
        return self

    def capture_method_mix_from_sim(self, sim, print_method_mix=False) -> 'MethodIntervention':
        """
        Capture the current method mix from a simulation to use as your baseline.
        
        **Why you need this:** When you want to change specific method shares (e.g., increase
        implants to 30%), you need to tell the tool what the current mix is so it knows how to
        adjust the other methods proportionally.
        
        Parameters
        ----------
        sim : fp.Sim
            An initialized (but not necessarily run) simulation object
            Must have called sim.init() before passing it here
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Typical workflow for method mix intervention
        >>> baseline_sim = fp.Sim(pars=pars)
        >>> baseline_sim.init()  # Must initialize first!
        >>> 
        >>> mod = MethodIntervention(year=2025)
        >>> mod.capture_method_mix_from_sim(baseline_sim)  # Capture baseline
        >>> mod.set_method_mix('impl', 0.25)  # Now set your target
        >>> 
        >>> # Build and run intervention simulation
        >>> intv = mod.build()
        >>> intervention_sim = fp.Sim(pars=pars, interventions=intv)
        >>> intervention_sim.run()
        
        Notes 
        --------------------
        - You must call sim.init() before using this function
        - This captures the method mix from the location's calibrated data
        - Use the same sim parameters for both baseline and intervention sims
        - The baseline sim can be discarded after capturing (you don't need to run it)
        """
        cm = sim.connectors.contraception
        mix = cm.pars.get('method_mix')
        if mix is None:
            raise ValueError('Simulation contraception module does not expose a method mix baseline')
        self.set_method_mix_baseline(mix)
        if print_method_mix:
            print(f'Method mix baseline: {mix}')
        return self

    def set_switching_matrix(self, matrix_or_path: Union[dict, str, Path]) -> 'MethodIntervention':
        """Set the switching matrix from a dict, file path, or Path object."""
        self._config.switch = load_config(matrix_or_path)
        return self

    def scale_switching_matrix(self, sim, target_method: str, scale_factor: float) -> 'MethodIntervention':
        """
        Modify the probability that people switch TO a specific method from other methods.
        
        **What this models:** Changes in method-switching behavior, which could result from:
        - Provider training on discussing method transitions
        - Improved access to specific methods at service points
        - Reduced barriers to switching (e.g., simplified counseling protocols)
        - Community education about alternative methods
        - Method-specific subsidies or promotions
        
        **Technical note:** This modifies the "switching matrix" - the probabilities that someone
        on Method A switches to Method B. You're increasing the probability that people switch
        TO your target method from all other methods.
        
        Parameters
        ----------
        sim : fp.Sim
            An initialized simulation (must have called sim.init())
        target_method : str
            Method to make more attractive for switching: 'inj', 'impl', 'iud', 'pill', etc.
        scale_factor : float
            Multiplier for switching probability. Examples:
            - 1.2 = 20% increase in probability of switching to this method
            - 1.5 = 50% increase
            - 0.8 = 20% decrease
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Model improved access to injectables at health facilities
        >>> baseline_sim = fp.Sim(pars=pars)
        >>> baseline_sim.init()
        >>> 
        >>> mod = MethodIntervention(year=2025)
        >>> mod.scale_switching_matrix(baseline_sim, 'inj', 1.3)  # 30% more likely to switch to injectables
        >>> mod.set_duration_months('inj', 36)  # Also improve continuation
        >>> 
        >>> intv = mod.build()
        >>> intervention_sim = fp.Sim(pars=pars, interventions=intv)
        >>> intervention_sim.run()
        
        How This Works
        --------------
        Imagine the baseline switching probabilities are:
        - From pills → injectables: 20%
        - From implants → injectables: 15%
        - From condoms → injectables: 10%
        
        With scale_factor=1.5 (50% increase), they become:
        - From pills → injectables: 30% (20% × 1.5)
        - From implants → injectables: 22.5% (15% × 1.5)
        - From condoms → injectables: 15% (10% × 1.5)
        
        All probabilities are automatically re-normalized to sum to 100%.
        
        Notes 
        --------------------
        - This is an ADVANCED feature - start with set_duration_months if you're new
        - Use this when you want to model improved access to switching, not initial uptake
        - Typically combine with duration changes for comprehensive interventions
        - Scale factors between 1.1-1.5 (10-50% increase) are typical for programmatic changes
        - The sim must be initialized first: sim.init()
        """
        # Normalize the method name
        target_name = self._normalize_name(target_method)
        
        # Get the contraception module
        cm = sim.connectors.contraception
        base = cm.pars.get('method_choice_pars')
        if base is None:
            raise ValueError('No method_choice_pars found in contraception module')
        
        new = sc.dcp(base)
        
        # Resolve destination method index position within the vector
        # Find the method by its name (which matches target_name)
        target_idx_number = None
        for method_short, method_obj in cm.methods.items():
            if method_short == target_name or (hasattr(method_obj, 'name') and method_obj.name == target_name):
                target_idx_number = method_obj.idx
                break
        
        if target_idx_number is None:
            raise ValueError(f'Could not find method "{target_name}" in sim contraception module')
        
        # Helper to compute destination position in the vector using 'method_idx' list
        def dest_pos(method_choice_part):
            idx_list = list(method_choice_part['method_idx'])
            try:
                return idx_list.index(target_idx_number)
            except ValueError:
                return None
        
        # Scale and re-normalize each origin/age bin
        for age_bin_key, age_bin_data in new.items():
            if not isinstance(age_bin_data, dict):
                continue
            for origin_key, origin_data in age_bin_data.items():
                if not isinstance(origin_data, dict):
                    continue
                if 'probs' not in origin_data or 'method_idx' not in origin_data:
                    continue
                
                pos = dest_pos(origin_data)
                if pos is None:
                    continue
                
                probs = np.array(origin_data['probs'], dtype=float)
                if pos >= len(probs):
                    continue
                
                # Scale the target probability
                probs[pos] *= scale_factor
                
                # Re-normalize
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                
                origin_data['probs'] = probs.tolist()
        
        self._config.switch = new
        return self

    def add_method(self, method, copy_from_row: str, copy_from_col: str,
                   initial_share: float = 0.1, renormalize: bool = True) -> 'MethodIntervention':
        """
        Add a new contraceptive method to the simulation.
        
        What this models:
        Introduction of a new contraceptive product or method that wasn't
        previously available. This could represent:
        - Launch of a new LARC product (e.g., subcutaneous DMPA)
        - Introduction of emergency contraception
        - New delivery mechanism for existing method (e.g., contraceptive patch)
        - Country-specific method introduction (e.g., first availability of implants)
        
        The new method is added by copying switching behavior from existing similar methods
        and expanding the switching matrix to accommodate the new option.
        
        Internally this constructs a :class:`fpsim.methods.NewMethodConfig`
        payload that is passed through to the core contraception module.
        
        Parameters
        ----------
        method : Method
            A Method object defining the new contraceptive method.
            Must include: name, label, efficacy, modern flag, duration distribution
        copy_from_row : str
            Method name to copy "switching FROM" behavior from (e.g., 'inj')
            This determines how people switch away from your new method
        copy_from_col : str
            Method name to copy "switching TO" behavior from (e.g., 'inj')
            This determines how people switch to your new method from other methods
        initial_share : float or callable or distribution, default=0.1
            Probability of staying on the new method in the switching matrix (0-1).
            You can provide a scalar, a callable returning a scalar, or a Starsim/scipy
            distribution object (e.g., ``ss.beta(...)``); it will be evaluated once
            when the intervention is built.
        renormalize : bool, default=True
            Whether to renormalize switching probabilities to sum to 1
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Example 1: Add subcutaneous DMPA (similar to regular injectables)
        >>> import fpsim as fp
        >>> 
        >>> # Define the new method
        >>> sc_dmpa = fp.Method(
        ...     name='sc_dmpa',
        ...     label='SC-DMPA',
        ...     efficacy=0.98,
        ...     modern=True,
        ...     dur_use=fp.methods.ln(4, 2.5),  # Lognormal: mean=4 months, std=2.5
        ...     csv_name='SC-DMPA'
        ... )
        >>> 
        >>> # Create intervention to add it in 2025
        >>> mod = fp.MethodIntervention(year=2025, label='Introduce SC-DMPA')
        >>> mod.add_method(
        ...     method=sc_dmpa,
        ...     copy_from_row='inj',  # Copy injectable switching patterns
        ...     copy_from_col='inj',
        ...     initial_share=0.12  # 12% stay on this method
        ... )
        >>> 
        >>> # Optionally adjust efficacy or duration after adding
        >>> mod.set_efficacy('sc_dmpa', 0.99)
        >>> mod.set_duration_months('sc_dmpa', 5)
        >>> 
        >>> intv = mod.build()
        >>> sim = fp.Sim(pars=pars, interventions=intv)
        >>> sim.run()
        
        >>> # Example 2: Add emergency contraception (similar to pills)
        >>> ec_method = fp.Method(
        ...     name='ec',
        ...     label='Emergency contraception',
        ...     efficacy=0.85,
        ...     modern=True,
        ...     dur_use=fp.methods.ln(0.5, 1),  # Very short duration
        ...     csv_name='EC'
        ... )
        >>> 
        >>> mod = fp.MethodIntervention(year=2023)
        >>> mod.add_method(
        ...     method=ec_method,
        ...     copy_from_row='pill',
        ...     copy_from_col='pill',
        ...     initial_share=0.05
        ... )
        
        How This Works
        --------------
        When you add a new method:
        1. The method is added to the available methods list
        2. The switching matrix is expanded by copying transition probabilities:
           - Rows (FROM new method): copied from copy_from_row method
           - Columns (TO new method): copied from copy_from_col method
        3. All transition probabilities are renormalized to sum to 100%
        4. The method becomes available for selection starting in the specified year
        
        Choosing Copy Methods
        ----------------------
        Choose copy_from_row and copy_from_col based on similarity:
        - **New LARC** → copy from 'impl' or 'iud'
        - **New short-acting** → copy from 'pill' or 'inj'
        - **New barrier method** → copy from 'cond'
        - **New permanent method** → copy from 'btl'
        
        Often you'll use the same method for both row and col (e.g., copy_from_row='inj'
        and copy_from_col='inj') if your new method is similar to an existing one.
        
        Notes 
        --------------------
        - Define the Method object with realistic efficacy and duration values
        - Use ln(mean, std) for duration: fp.methods.ln(24, 3) = 24 months average
        - The new method will appear in results using the 'label' you provide
        - Choose similar methods for copying to get realistic switching patterns
        - You can combine add_method with other modifications (efficacy, duration, etc.)
        - The method becomes available simulation-wide starting in the intervention year
        """
        # Import here to avoid circular dependency
        from . import methods as fpm
        
        # Validate method object
        if not isinstance(method, fpm.Method):
            raise ValueError('method must be a fpsim.Method object')
        
        if method.name is None or method.label is None:
            raise ValueError('Method must have both name and label defined')
        
        if method.efficacy is None or method.dur_use is None:
            raise ValueError('Method must have efficacy and dur_use defined')
        
        # Normalise reference method names (must exist already)
        normalized_row = self._normalize_name(copy_from_row)
        normalized_col = self._normalize_name(copy_from_col)

        # Store the new method configuration as a dataclass payload
        self._config.new_method = NewMethodConfig(
            method=method,
            copy_from_row=normalized_row,
            copy_from_col=normalized_col,
            initial_share=initial_share,
            renormalize=renormalize,
        )

        return self

    # -----------------
    # Introspection
    # -----------------
    def preview(self) -> PreviewDict:
        """
        Preview the intervention configuration before building it.
        
        Shows you exactly what changes will be applied in a readable format.
        This is useful for checking your intervention setup before running the simulation.
        
        Returns
        -------
        dict
            Summary of all configured changes including:
            - year: When the intervention starts
            - efficacy: Method efficacy changes (if any)
            - duration_months: Duration changes (if any)
            - p_use: Probability of use change (if any)
            - method_mix: Method mix targets (if any)
            - switching_matrix: Whether switching matrix was modified
            - label: Intervention label
            
        Examples
        --------
        >>> mod = MethodIntervention(year=2025, label='Counseling Program')
        >>> mod.set_efficacy('inj', 0.985)
        >>> mod.set_duration_months('inj', 36)
        >>> 
        >>> # Preview before building
        >>> print(mod.preview())
        {'year': 2025,
         'efficacy': {'inj': 0.985},
         'duration_months': {'inj': 36},
         'p_use': None,
         'method_mix': None,
         'switching_matrix': None,
         'label': 'Counseling Program'}
        
        Notes 
        --------------------
        - Use this to verify your intervention before running (catches typos/mistakes)
        - Check that method names are correct and values make sense
        - None means that parameter wasn't modified (uses baseline values)
        """
        summary: PreviewDict = {
            'year': self.year,
            'efficacy': sc.dcp(self._config.eff) or None,
            'duration_months': sc.dcp(self._config.dur) or None,
            'p_use': self._config.p_use,
            'method_mix': (
                None if not self._config.mix_values
                else {
                    name: val
                    for name, val in zip(self.method_mix_order, self._build_method_mix_array())
                }
            ),
            'switching_matrix': (None if self._config.switch is None else '<dict>'),
            'new_method': (
                None
                if self._config.new_method is None
                else {
                    'name': self._config.new_method.method.name,
                    'label': self._config.new_method.method.label,
                    'efficacy': self._config.new_method.method.efficacy,
                    'copy_from_row': self._config.new_method.copy_from_row,
                    'copy_from_col': self._config.new_method.copy_from_col,
                    'initial_share': self._config.new_method.initial_share,
                }
            ),
            'label': self.label,
        }
        return summary

    # -----------------
    # Build
    # -----------------
    def build(self):
        """
        Build the intervention and return it ready to use in a simulation.
        
        This is the final step after configuring all your changes. It creates the
        actual intervention object that you pass to fp.Sim().
        
        Returns
        -------
        update_methods intervention
            An FPsim intervention ready to use in a simulation
            
        Examples
        --------
        >>> # Complete workflow
        >>> mod = MethodIntervention(year=2025, label='Injectable Program')
        >>> mod.set_efficacy('inj', 0.99)
        >>> mod.set_duration_months('inj', 36)
        >>> 
        >>> # Build the intervention
        >>> intv = mod.build()
        >>> 
        >>> # Use it in a simulation
        >>> sim = fp.Sim(pars=pars, interventions=intv)
        >>> sim.run()
        
        Notes 
        --------------------
        - Always call build() as the last step after all set_X() calls
        - The returned intervention can be used directly in fp.Sim()
        - You can pass multiple interventions as a list: fp.Sim(interventions=[intv1, intv2])
        - Use preview() before build() to check your configuration
        """
        # Convert method names to standard labels for the intervention
        # For dynamically added methods, use the method name as-is
        if self._config.eff:
            label_lookup = self._name_to_label.get
            eff_by_label = {label_lookup(name, name): val for name, val in self._config.eff.items()}
        else:
            eff_by_label = None
            
        if self._config.dur:
            label_lookup = self._name_to_label.get
            dur_by_label = {label_lookup(name, name): val for name, val in self._config.dur.items()}
        else:
            dur_by_label = None
        
        kwargs = dict(
            year=self.year,
            eff=eff_by_label,
            dur_use=dur_by_label,
            p_use=self._config.p_use,
            method_mix=self._build_method_mix_array(),
            method_choice_pars=self._config.switch,
            new_method=self._config.new_method,
        )
        # Drop None values to keep payload clean
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Pass label as name to make intervention unique
        if self.label:
            kwargs['name'] = self.label
            
        return fpi.update_methods(**kwargs)


def make_update_methods(
    year: float,
    method: Optional[str] = None,
    efficacy: Optional[float] = None,
    duration_months: Optional[Union[int, float]] = None,
    p_use: Optional[float] = None,
    method_mix_value: Optional[float] = None,
    method_mix_baseline: Optional[Sequence[float]] = None,
    baseline_sim=None,
    switch: Optional[Union[dict, str]] = None,
    label: Optional[str] = None,
):
    """
    Functional shortcut to build an update_methods intervention for a single method.
    
    Parameters
    ----------
    year : float
        Year when intervention takes effect
    method : str, optional
        Method name for efficacy, duration, or method_mix changes
    efficacy : float, optional
        New efficacy value (0-1)
    duration_months : int or float, optional
        New duration in months
    p_use : float, optional
        Probability of contraceptive use (0-1)
    method_mix_value : float, optional
        Target method mix value for the specified method
    method_mix_baseline : Sequence[float], optional
        Explicit baseline method mix values (9 methods)
    baseline_sim : fp.Sim, optional
        Baseline simulation to auto-capture method mix from (alternative to method_mix_baseline)
    switch : dict or str, optional
        Switching matrix as dict or path to file
    label : str, optional
        Label for the intervention
        
    Returns
    -------
    update_methods
        Built intervention ready for simulation
        
    Examples
    --------
    >>> # Simple efficacy change
    >>> intv = make_update_methods(year=2025, method='pill', efficacy=0.95)
    
    >>> # Method mix with auto-capture
    >>> baseline_sim = fp.Sim(location='kenya')
    >>> baseline_sim.init()
    >>> intv = make_update_methods(s
    ...     year=2025, 
    ...     method='impl', 
    ...     method_mix_value=0.30,
    ...     baseline_sim=baseline_sim
    ... )
    """
    builder = MethodIntervention(year=year, label=label)
    
    # Handle baseline - explicit array or auto-capture from sim
    if method_mix_baseline is not None:
        builder.set_method_mix_baseline(method_mix_baseline)
    elif baseline_sim is not None:
        builder.capture_method_mix_from_sim(baseline_sim)
    
    if efficacy is not None:
        if method is None:
            raise ValueError('`method` must be provided when setting efficacy')
        builder.set_efficacy(method, efficacy)
    if duration_months is not None:
        if method is None:
            raise ValueError('`method` must be provided when setting duration_months')
        builder.set_duration_months(method, duration_months)
    if p_use is not None:
        builder.set_probability_of_use(p_use)
    if method_mix_value is not None:
        if method is None:
            raise ValueError('`method` must be provided when setting method_mix_value')
        builder.set_method_mix(method, method_mix_value)
    if switch is not None:
        builder.set_switching_matrix(switch)
    return builder.build()

