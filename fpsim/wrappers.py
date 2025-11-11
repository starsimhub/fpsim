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

Key concepts for PST teams:
---------------------------
- **Efficacy**: How well the method prevents pregnancy (0 to 1, where 0.99 = 99% effective)
- **Duration**: How many months people typically stay on the method before discontinuing
- **Method mix**: The distribution of users across different contraceptive methods
- **Switching matrix**: The probabilities of moving from one method to another

Note: Small changes in duration can have large effects on method mix because people
accumulate on methods they stay on longer. This is realistic - improving continuation
rates is one of the most impactful interventions.

Quick Reference for PST Teams
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
from typing import Dict, Optional, Sequence, Union

import numpy as np
import sciris as sc

from . import interventions as fpi

__all__ = [
    'MethodIntervention',
    'make_update_methods',
]


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
    
    Important Notes for PST Teams
    ------------------------------
    - **Duration impacts**: Duration changes translate directly to usage changes in steady state.
      Increasing duration by 50% increases method usage by 50% because people accumulate on
      the method over time. This is realistic and expected!
    - **Method competition**: When one method improves, similar methods often decrease
      (e.g., improving injectables may decrease implants as they compete for users)
    - **Units**: Efficacy is 0-1 (e.g., 0.95 = 95%), Duration is in months (e.g., 24 = 2 years)
    - **Mix values**: Can provide as percentages (25) or fractions (0.25), both work
    """

    # Method names (must match method.name in fpsim)
    _method_names = [
        'none',
        'pill',
        'iud',
        'inj',
        'cond',
        'btl',
        'wdraw',
        'impl',
        'othtrad',
        'othmod',
    ]
    
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

        # Internal payload pieces
        self._eff: Dict[str, float] = {}
        self._dur: Dict[str, Union[int, float]] = {}
        self._p_use: Optional[float] = None
        self._mix_values: Dict[str, float] = {}
        self._method_mix_base: Optional[np.ndarray] = None
        self._switch: Optional[dict] = None
        self._new_method: Optional[dict] = None

        self._method_mix_order = [name for name in self._method_names if name != 'none']

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
        if path.lower().endswith(('.json',)):
            with open(path, 'r') as f:
                return json.load(f)
        if path.lower().endswith(('.yml', '.yaml')):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ImportError('YAML support not available. Please install pyyaml to load YAML files.') from e
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        raise ValueError('Unsupported file type for switching matrix; expected .json or .yaml')

    def _convert_mix_value(self, value: float) -> float:
        val = float(value)
        if val < 0:
            raise ValueError('Method mix entries must be non-negative')
        if val > 1.0 + 1e-6:
            val = val / 100.0
        return val

    def _ensure_method_mix_base(self) -> np.ndarray:
        if self._method_mix_base is None:
            raise ValueError(
                'Method mix baseline not set. Call `set_method_mix_baseline()` or '
                '`capture_method_mix_from_sim(sim)` before modifying the mix.'
            )
        return self._method_mix_base

    def _build_method_mix_array(self) -> Optional[np.ndarray]:
        if not self._mix_values:
            return None

        base = self._ensure_method_mix_base()
        arr = base.copy()

        targeted_indices = set()
        targeted_sum = 0.0
        for name, val in self._mix_values.items():
            if name == 'none':
                raise ValueError('Method mix cannot be set for "none"; use probability of use instead.')
            if name not in self._method_mix_order:
                raise ValueError(f'Method mix updates must reference one of: {self._method_mix_order}')
            idx = self._method_mix_order.index(name)
            new_val = self._convert_mix_value(val)
            arr[idx] = new_val
            targeted_indices.add(idx)
            targeted_sum += new_val

        if targeted_sum > 1.0 + 1e-6:
            raise ValueError('Method mix updates sum to more than 1.0; reduce the requested values.')

        base_remaining = base.copy()
        base_remaining[list(targeted_indices)] = 0.0
        remaining_total_base = base_remaining.sum()

        if remaining_total_base < 1e-12:
            if targeted_sum < 1.0 - 1e-6:
                raise ValueError('Baseline for non-targeted methods sums to zero; cannot redistribute remaining probability.')
            scale = 0.0
        else:
            scale = max(0.0, (1.0 - targeted_sum) / remaining_total_base)

        for idx, base_val in enumerate(base):
            if idx in targeted_indices:
                continue
            arr[idx] = base_val * scale

        total = arr.sum()
        if total <= 0:
            raise ValueError('Provided method mix sums to zero after applying updates')
        return arr / total

    # -----------------
    # Builder methods
    # -----------------
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
        
        Notes for PST Teams
        --------------------
        - Typical efficacies: Pills ~94%, IUDs ~99%, Implants ~99%, Condoms ~95%
        - Small changes (2-3%) can have meaningful impact on unintended pregnancies
        - Efficacy changes affect pregnancy outcomes but don't directly change method mix
        """
        method_name = self._normalize_name(method)
        val = float(efficacy)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f'Efficacy for {method_name} must be between 0 and 1')
        self._eff[method_name] = val
        if print_efficacy:
            print(f'Efficacy: {method_name} = {val}')
        return self

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
        
        Critical Notes for PST Teams
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
        method_name = self._normalize_name(method)
        mval = float(months)
        if mval <= 0:
            raise ValueError(f'Duration (months) for {method_name} must be positive')
        self._dur[method_name] = mval
        if print_duration:
            print(f'Duration: {method_name} = {mval} months')
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
        
        Notes for PST Teams
        --------------------
        - This affects overall contraceptive prevalence (CPR)
        - Use this for interventions that increase access/demand broadly
        - For method-specific interventions, use set_duration_months or set_method_mix instead
        - Note: May not work with all FPsim choice modules (the model will warn you)
        """
        p = float(p_use)
        if not (0.0 <= p <= 1.0):
            raise ValueError('Probability of use must be between 0 and 1')
        self._p_use = p
        if print_p_use:
            print(f'Probability of use: {p}')
        return self

    def set_method_mix(self, method: str, value: float, print_method_mix=False) -> 'MethodIntervention':
        """
        Set the target share/percentage for a specific contraceptive method.
        
        **What this models:** Changes in which methods people choose (method mix), which 
        could result from:
        - Social marketing campaigns promoting specific methods
        - Provider training on method-specific counseling
        - Introduction of new methods or brands
        - Changes in method availability at service delivery points
        - Community-based distribution focusing on specific methods
        
        **IMPORTANT:** You must call `capture_method_mix_from_sim()` first to establish
        a baseline, so other methods retain their current shares and adjust proportionally.
        
        Parameters
        ----------
        method : str
            Method name: 'pill', 'iud', 'inj', 'impl', 'cond', etc.
        value : float
            Target share as decimal (0.25 = 25%) or percentage (25 = 25%)
            The tool will auto-detect and normalize either format
            
        Returns
        -------
        self : MethodIntervention
            Returns self for method chaining
            
        Examples
        --------
        >>> # Model LARC promotion campaign targeting 30% implant use
        >>> baseline_sim = fp.Sim(pars=pars)
        >>> baseline_sim.init()
        >>> mod = MethodIntervention(year=2025)
        >>> mod.capture_method_mix_from_sim(baseline_sim)  # Capture current mix
        >>> mod.set_method_mix('impl', 0.30)  # Target 30% implants
        
        >>> # Model multi-method social marketing
        >>> mod.capture_method_mix_from_sim(baseline_sim)
        >>> mod.set_method_mix('inj', 0.25)  # Target 25% injectables
        >>> mod.set_method_mix('impl', 0.20)  # Target 20% implants
        >>> # Other methods will be rescaled proportionally to sum to 100%
        
        How Method Mix Adjustment Works
        --------------------------------
        1. You set target shares for one or more methods (e.g., implants = 30%)
        2. Other methods keep their relative proportions but scale down to make room
        3. Everything sums to 100% automatically
        
        Example: If baseline is [Pills: 20%, Injectables: 40%, Implants: 40%]
        And you set implants to 50%, result will be:
        [Pills: 10%, Injectables: 20%, Implants: 50%]
        (Pills and injectables scaled down proportionally)
        
        Notes for PST Teams
        --------------------
        - Always capture baseline first: `mod.capture_method_mix_from_sim(baseline_sim)`
        - You can target multiple methods in one intervention
        - Non-targeted methods adjust automatically (proportional rescaling)
        - Use percentages (25) or decimals (0.25), both work
        - Method mix changes model demand-side shifts, not supply constraints
        """
        method_name = self._normalize_name(method)
        val = float(value)
        if val < 0:
            raise ValueError(f'Method mix for {method_name} cannot be negative')
        self._mix_values[method_name] = val
        if print_method_mix:
            print(f'Method mix: {method_name} = {val}')
        return self

    def set_method_mix_baseline(self, mix_values: Sequence[float]) -> 'MethodIntervention':
        """Store the baseline method mix so that only requested entries are updated."""
        arr = np.asarray(mix_values, dtype=float)
        if arr.ndim != 1:
            raise ValueError('Baseline method mix must be a one-dimensional sequence')
        if arr.size != len(self._method_mix_order):
            raise ValueError(f'Baseline method mix must have {len(self._method_mix_order)} entries')
        if (arr < 0).any():
            raise ValueError('Baseline method mix cannot contain negative values')
        if arr.max() > 1.0 + 1e-6:
            arr = arr / 100.0
        total = arr.sum()
        if total <= 0:
            raise ValueError('Baseline method mix sums to zero')
        self._method_mix_base = arr / total
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
        
        Notes for PST Teams
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

    def set_switching_matrix(self, matrix_or_path: Union[dict, str]) -> 'MethodIntervention':
        if isinstance(matrix_or_path, str):
            self._switch = self._load_file(matrix_or_path)
        elif isinstance(matrix_or_path, dict):
            self._switch = sc.dcp(matrix_or_path)
        else:
            raise ValueError('Switching matrix must be a dict or a path to a JSON/YAML file')
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
        
        Notes for PST Teams
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
        
        self._switch = new
        return self

    def add_method(self, method, copy_from_row: str, copy_from_col: str,
                   initial_share: float = 0.1, renormalize: bool = True) -> 'MethodIntervention':
        """
        Add a new contraceptive method to the simulation.
        
        **What this models:** Introduction of a new contraceptive product or method that wasn't
        previously available. This could represent:
        - Launch of a new LARC product (e.g., subcutaneous DMPA)
        - Introduction of emergency contraception
        - New delivery mechanism for existing method (e.g., contraceptive patch)
        - Country-specific method introduction (e.g., first availability of implants)
        
        The new method is added by copying switching behavior from existing similar methods
        and expanding the switching matrix to accommodate the new option.
        
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
        initial_share : float, default=0.1
            Probability of staying on the new method in the switching matrix (0-1)
            Higher values mean people are more likely to continue the new method
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
        
        Notes for PST Teams
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
        
        # Validate copy methods
        copy_from_row = self._normalize_name(copy_from_row)
        copy_from_col = self._normalize_name(copy_from_col)
        
        # Validate initial_share
        if not (0.0 <= initial_share <= 1.0):
            raise ValueError('initial_share must be between 0 and 1')
        
        # Store the new method configuration
        self._new_method = {
            'method': method,
            'copy_from_row': copy_from_row,
            'copy_from_col': copy_from_col,
            'initial_share': initial_share,
            'renormalize': renormalize
        }
        
        return self

    # -----------------
    # Introspection
    # -----------------
    def preview(self) -> dict:
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
        
        Notes for PST Teams
        --------------------
        - Use this to verify your intervention before running (catches typos/mistakes)
        - Check that method names are correct and values make sense
        - None means that parameter wasn't modified (uses baseline values)
        """
        summary = dict(
            year=self.year,
            efficacy=sc.dcp(self._eff) or None,
            duration_months=sc.dcp(self._dur) or None,
            p_use=self._p_use,
            method_mix=(
                None if not self._mix_values
                else {
                    name: val
                    for name, val in zip(self._method_mix_order, self._build_method_mix_array())
                }
            ),
            switching_matrix=(None if self._switch is None else '<dict>'),
            new_method=(
                None if self._new_method is None 
                else {
                    'name': self._new_method['method'].name,
                    'label': self._new_method['method'].label,
                    'efficacy': self._new_method['method'].efficacy,
                    'copy_from_row': self._new_method['copy_from_row'],
                    'copy_from_col': self._new_method['copy_from_col'],
                    'initial_share': self._new_method['initial_share'],
                }
            ),
            label=self.label,
        )
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
        
        Notes for PST Teams
        --------------------
        - Always call build() as the last step after all set_X() calls
        - The returned intervention can be used directly in fp.Sim()
        - You can pass multiple interventions as a list: fp.Sim(interventions=[intv1, intv2])
        - Use preview() before build() to check your configuration
        """
        # Convert method names to labels for the intervention
        # For dynamically added methods, use the name as-is if no label mapping exists
        eff_by_label = {self._name_to_label.get(name, name): val for name, val in self._eff.items()} if self._eff else None
        dur_by_label = {self._name_to_label.get(name, name): val for name, val in self._dur.items()} if self._dur else None
        
        kwargs = dict(
            year=self.year,
            eff=eff_by_label,
            dur_use=dur_by_label,
            p_use=self._p_use,
            method_mix=self._build_method_mix_array(),
            method_choice_pars=self._switch,
            new_method=self._new_method,
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
    switch: Optional[Union[dict, str]] = None,
    label: Optional[str] = None,
):
    """
    Functional shortcut to build an update_methods intervention for a single method.
    """
    builder = MethodIntervention(year=year, label=label)
    if method_mix_baseline is not None:
        builder.set_method_mix_baseline(method_mix_baseline)
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

