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
    >>> mod.set_duration_months('inj', 36)  # People stay on injectables for 3 years instead of 2
    
    **Social marketing campaign shifts method mix:**
    >>> mod = MethodIntervention(year=2025)
    >>> mod.capture_method_mix_from_sim(baseline_sim)
    >>> mod.set_method_mix('impl', 0.25)  # Target 25% of users choosing implants
    
    Important Notes for PST Teams
    ------------------------------
    - **Duration impacts**: Increasing duration by 50% can more than double method usage
      because people accumulate on the method over time. This is realistic!
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

        self._method_mix_order = [name for name in self._method_names if name != 'none']

    # -----------------
    # Helper utilities
    # -----------------
    def _normalize_name(self, name: str) -> str:
        """Validate that method name is in the approved list."""
        if not isinstance(name, str):
            raise ValueError('Method name must be a string')
        if name not in self._method_names:
            raise ValueError(f'Method name must be one of: {self._method_names}')
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
    def set_efficacy(self, method: str, efficacy: float) -> 'MethodIntervention':
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
        return self

    def set_duration_months(self, method: str, months: Union[int, float]) -> 'MethodIntervention':
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
        >>> mod.set_duration_months('inj', 36)  # Increase from ~24 to 36 months
        
        >>> # Model supply chain improvements reducing pill stockouts
        >>> mod.set_duration_months('pill', 18)  # Increase from ~12 to 18 months
        
        >>> # Model comprehensive program improving multiple methods
        >>> mod.set_duration_months('inj', 36).set_duration_months('impl', 48)
        
        Critical Notes for PST Teams
        -----------------------------
        **Duration changes have LARGE effects on method mix:**
        - Increasing duration by 50% can MORE than double method usage in steady state
        - This is because people "accumulate" on methods they stay on longer
        - Example: If 100 people/month start injectables:
          * At 24 months average duration → ~2,400 current users at any time
          * At 36 months average duration → ~3,600 current users at any time
          * That's a 50% increase in snapshot prevalence!
        
        **Why this is realistic:**
        - Duration/continuation is one of the most impactful real-world program interventions
        - Studies show counseling can increase continuation by 30-60%
        - The large method mix changes you see in results are expected, not bugs!
        
        **Typical baseline durations:**
        - Pills: 12-18 months
        - Injectables: 18-24 months  
        - IUDs: 36-60 months
        - Implants: 36-48 months
        - Condoms: 6-12 months (sporadic use)
        """
        method_name = self._normalize_name(method)
        mval = float(months)
        if mval <= 0:
            raise ValueError(f'Duration (months) for {method_name} must be positive')
        self._dur[method_name] = mval
        return self

    def set_probability_of_use(self, p_use: float) -> 'MethodIntervention':
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
        return self

    def set_method_mix(self, method: str, value: float) -> 'MethodIntervention':
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

    def capture_method_mix_from_sim(self, sim) -> 'MethodIntervention':
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
        eff_by_label = {self._name_to_label[name]: val for name, val in self._eff.items()} if self._eff else None
        dur_by_label = {self._name_to_label[name]: val for name, val in self._dur.items()} if self._dur else None
        
        kwargs = dict(
            year=self.year,
            eff=eff_by_label,
            dur_use=dur_by_label,
            p_use=self._p_use,
            method_mix=self._build_method_mix_array(),
            method_choice_pars=self._switch,
        )
        # Drop None values to keep payload clean
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
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

