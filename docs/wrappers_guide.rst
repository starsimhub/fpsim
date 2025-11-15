=======================================
Contraceptive Interventions
=======================================

.. currentmodule:: fpsim

Overview
========

The ``fpsim.MethodIntervention`` feature provides an easy-to-use interface for modeling contraceptive interventions without needing to understand the complex internal FPsim structure. It's designed for program teams and researchers who want to model realistic family planning interventions.

What You Can Model
==================

1. **Efficacy changes**: Model improved method effectiveness (e.g., better quality pills)
2. **Duration changes**: Model how long people stay on a method (e.g., better counseling, reduced side effects)
3. **Method mix**: Model shifts in which methods people choose (e.g., social marketing campaigns)
4. **Probability of use**: Model changes in overall contraceptive uptake
5. **Switching patterns**: Model how people transition between methods
6. **New methods**: Add new contraceptive products to the simulation dynamically

Key Concepts
============

- **Efficacy**: How well the method prevents pregnancy (0 to 1, where 0.99 = 99% effective)
- **Duration**: How many months people typically stay on the method before discontinuing
- **Method mix**: The distribution of users across different contraceptive methods
- **Switching matrix**: The probabilities of moving from one method to another

.. note::
   Small changes in duration can have **large effects** on method mix because people accumulate on methods they stay on longer. This is realistic - improving continuation rates is one of the most impactful interventions.

Quick Reference
===============

**What intervention should I model for...**

1. **Better counseling / side effect management** → :meth:`MethodIntervention.set_duration_months`
   
   People stay on methods longer with better support

2. **Improved product quality** → :meth:`MethodIntervention.set_efficacy`
   
   Better manufacturing or user education increases effectiveness

3. **Social marketing campaign for specific method** → :meth:`MethodIntervention.set_method_mix`
   
   Shift demand toward targeted methods (requires baseline capture)

4. **General awareness campaign** → :meth:`MethodIntervention.set_probability_of_use`
   
   Increase overall contraceptive use (CPR) - **Only works with RandomChoice module**

5. **Provider training on method transitions** → :meth:`MethodIntervention.scale_switching_matrix`
   
   Make it easier for people to switch to better methods

6. **New product introduction** → :meth:`MethodIntervention.add_method`
   
   Add new contraceptive methods during simulation

7. **Comprehensive program** → Combine multiple methods!
   
   Example: Better counseling (duration) + quality improvements (efficacy)

Method Name Shortcuts
=====================

Use these short names when calling the intervention methods:

- ``'pill'`` = Pills
- ``'inj'`` = Injectables (DMPA, Sayana Press, etc.)
- ``'impl'`` = Implants (Jadelle, Implanon, etc.)
- ``'iud'`` = IUDs
- ``'cond'`` = Male condoms
- ``'btl'`` = Female sterilization (tubal ligation)
- ``'wdraw'`` = Withdrawal
- ``'othtrad'`` = Other traditional methods
- ``'othmod'`` = Other modern methods

Classes
=======

MethodIntervention
------------------

.. autoclass:: MethodIntervention
   :members:
   :undoc-members:
   :show-inheritance:

   The main class for building contraceptive method interventions.

   **Basic Usage Pattern:**

   1. Create a modifier: ``mod = fp.MethodIntervention(year=2025)``
   2. Configure changes: ``mod.set_efficacy('pill', 0.95).set_duration_months('pill', 18)``
   3. Build intervention: ``intv = mod.build()``
   4. Run simulation: ``sim = fp.Sim(pars=pars, interventions=intv)``

   **Real-World Examples:**

   Improving method quality::

       mod = fp.MethodIntervention(year=2025)
       mod.set_efficacy('pill', 0.95)  # Improve pill effectiveness to 95%

   Better counseling increases continuation::

       mod = fp.MethodIntervention(year=2025)
       mod.set_duration_months('inj', 36)  # Set target duration to 36 months

   Social marketing campaign shifts method mix::

       mod = fp.MethodIntervention(year=2025)
       mod.capture_method_mix_from_sim(baseline_sim)
       mod.set_method_mix('impl', 0.25)  # Target 25% of users choosing implants

   Adding a new method::

       sc_dmpa = fp.Method(
           name='sc_dmpa',
           label='SC-DMPA',
           efficacy=0.98,
           modern=True,
           dur_use=fp.methods.ln(4, 2.5)
       )
       mod = fp.MethodIntervention(year=2025, label='Introduce SC-DMPA')
       mod.add_method(
           method=sc_dmpa,
           copy_from_row='inj',
           copy_from_col='inj',
           initial_share=0.12
       )

MethodName
----------

.. autoclass:: MethodName
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of standard contraceptive method names.

Functions
=========

make_update_methods
-------------------

.. autofunction:: make_update_methods

Functional shortcut to build an intervention for a single method.

**Examples:**

Simple efficacy change::

    intv = fp.make_update_methods(year=2025, method='pill', efficacy=0.95)

Method mix with auto-capture::

    baseline_sim = fp.Sim(location='kenya')
    baseline_sim.init()
    intv = fp.make_update_methods(
        year=2025, 
        method='impl', 
        method_mix_value=0.30,
        baseline_sim=baseline_sim
    )

Complete Examples
=================

Example 1: Simple Efficacy Improvement
---------------------------------------

Model improved injectable quality::

    import fpsim as fp

    # Define parameters
    pars = dict(
        n_agents=5000,
        location='kenya',
        start_year=2000,
        end_year=2020
    )

    # Create intervention
    mod = fp.MethodIntervention(year=2010, label='Injectable Quality Program')
    mod.set_efficacy('inj', 0.99)  # Improve to 99% effectiveness

    # Build and run
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv)
    sim.run()

Example 2: LARC Promotion Program
----------------------------------

Model a comprehensive implant promotion program::

    import fpsim as fp

    pars = dict(
        n_agents=5000,
        location='senegal',
        start_year=2000,
        end_year=2020
    )

    # Baseline sim to capture current mix
    baseline_sim = fp.Sim(pars=pars)
    baseline_sim.init()

    # Create comprehensive program
    mod = fp.MethodIntervention(year=2010, label='Implant Promotion')
    mod.set_method_mix('impl', 0.25, baseline_sim=baseline_sim)  # Target 25%
    mod.set_duration_months('impl', 48)  # Better continuation (4 years)
    mod.set_efficacy('impl', 0.995)  # Slight quality improvement

    # Build and run
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv)
    sim.run()

Example 3: Adding a New Method
-------------------------------

Model the introduction of SC-DMPA (self-injectable)::

    import fpsim as fp

    pars = dict(
        n_agents=10000,
        location='kenya',
        start_year=2000,
        end_year=2020
    )

    # Define the new method
    sc_dmpa = fp.Method(
        name='sc_dmpa',
        label='SC-DMPA',
        efficacy=0.98,
        modern=True,
        dur_use=fp.methods.ln(4, 2.5),
        csv_name='SC-DMPA'
    )

    # Create intervention
    mod = fp.MethodIntervention(year=2015, label='Introduce SC-DMPA')
    mod.add_method(
        method=sc_dmpa,
        copy_from_row='inj',  # Copy switching from injectables
        copy_from_col='inj',
        initial_share=0.15  # 15% staying probability
    )
    
    # Optional: Modify after adding
    mod.set_efficacy('sc_dmpa', 0.99)
    mod.set_duration_months('sc_dmpa', 6)

    # Build and run
    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv)
    sim.run()

Example 4: Multiple New Methods
--------------------------------

Model introducing two methods at different times::

    import fpsim as fp

    pars = dict(
        n_agents=10000,
        location='kenya',
        start_year=2000,
        end_year=2020
    )

    # First method: SC-DMPA (2010)
    sc_dmpa = fp.Method(
        name='sc_dmpa',
        label='SC-DMPA',
        efficacy=0.94,
        modern=True,
        dur_use=fp.methods.ln(4, 2.5)
    )

    intv1 = fp.MethodIntervention(year=2010, label='Introduce SC-DMPA')
    intv1.add_method(sc_dmpa, 'inj', 'inj', initial_share=0.25)
    intv1.set_duration_months('sc_dmpa', 6)

    # Second method: Contraceptive Ring (2015)
    ring = fp.Method(
        name='ring',
        label='Contraceptive Ring',
        efficacy=0.91,
        modern=True,
        dur_use=fp.methods.ln(12, 3)
    )

    intv2 = fp.MethodIntervention(year=2015, label='Introduce Ring')
    intv2.add_method(ring, 'pill', 'pill', initial_share=0.20)
    intv2.set_duration_months('ring', 18)

    # Run with both interventions
    sim = fp.Sim(
        pars=pars, 
        interventions=[intv1.build(), intv2.build()]
    )
    sim.run()

Important Notes
===============

Duration Changes Have Large Effects
------------------------------------

Duration changes translate **directly** to usage changes in steady state. This is because people "accumulate" on methods they stay on longer.

**Example:** If 100 people/month start injectables:

- At 24 months average duration → ~2,400 current users at any time
- At 36 months average duration → ~3,600 current users at any time
- That's a 50% duration increase → 50% usage increase!

This is **realistic and expected** - continuation rate improvements are one of the most impactful real-world program interventions.

Probability of Use Limitations
-------------------------------

:meth:`MethodIntervention.set_probability_of_use` **only works with RandomChoice** contraception module.

.. warning::
   For ``SimpleChoice`` and ``StandardChoice`` modules, probability is calculated from individual attributes (age, education, wealth, parity) and **cannot be overridden** with a simple value.

**For SimpleChoice/StandardChoice, use instead:**

- Duration changes: ``mod.set_duration_months('inj', 36)``
- Method mix changes: ``mod.set_method_mix('impl', 0.25)``
- The ``change_initiation`` intervention class

See the `technical explanation <https://github.com/fpsim/fpsim/blob/main/MARKDOWNS/WHY_P_USE_ONLY_WORKS_WITH_RANDOMCHOICE.md>`_ for more details.

Method Mix Requires Baseline
-----------------------------

When using :meth:`MethodIntervention.set_method_mix`, you must provide a baseline either:

1. Passing ``baseline_sim`` parameter::

       mod.set_method_mix('impl', 0.30, baseline_sim=baseline_sim)

2. Or calling ``capture_method_mix_from_sim()`` first::

       mod.capture_method_mix_from_sim(baseline_sim)
       mod.set_method_mix('impl', 0.30)

Method Chaining
---------------

All setter methods return ``self`` for convenient chaining::

    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('inj', 0.99) \\
       .set_duration_months('inj', 36) \\
       .set_method_mix('inj', 0.40, baseline_sim=sim)

    intv = mod.build()

See Also
========

- :doc:`/tutorials/T3_interventions_methods` - Tutorial on interventions
- :doc:`/tutorials/T5_new_method` - Tutorial on adding new methods
- :mod:`fpsim.interventions` - Low-level intervention classes
- :mod:`fpsim.methods` - Contraceptive method definitions

