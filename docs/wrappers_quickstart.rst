=============================
 Quick Start Guide
=============================

.. currentmodule:: fpsim

This is a quick start guide for using the ``MethodIntervention`` wrapper. For detailed documentation, see :doc:`wrappers_guide`.

Basic Pattern
=============

**3 Steps to Create an Intervention:**

1. **Create** the modifier::

    mod = fp.MethodIntervention(year=2025, label='My Program')

2. **Configure** changes::

    mod.set_efficacy('inj', 0.99)
    mod.set_duration_months('inj', 36)

3. **Build** and use::

    intv = mod.build()
    sim = fp.Sim(pars=pars, interventions=intv)
    sim.run()

Common Scenarios
================

Improve Method Quality
----------------------

::

    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('pill', 0.95)
    intv = mod.build()

Increase Continuation (Better Counseling)
------------------------------------------

::

    mod = fp.MethodIntervention(year=2025)
    mod.set_duration_months('inj', 36)  # 36 months average use
    intv = mod.build()

Shift Method Mix (Social Marketing)
------------------------------------

::

    # Need baseline first
    baseline = fp.Sim(location='kenya')
    baseline.init()
    
    mod = fp.MethodIntervention(year=2025)
    mod.set_method_mix('impl', 0.30, baseline_sim=baseline)  # Target 30%
    intv = mod.build()

Add New Method
--------------

::

    # Define new method
    sc_dmpa = fp.Method(
        name='sc_dmpa',
        label='SC-DMPA',
        efficacy=0.98,
        modern=True,
        dur_use=fp.methods.ln(4, 2.5)
    )
    
    # Add it
    mod = fp.MethodIntervention(year=2015)
    mod.add_method(sc_dmpa, copy_from_row='inj', copy_from_col='inj', initial_share=0.15)
    intv = mod.build()

Combined Program
----------------

::

    mod = fp.MethodIntervention(year=2025, label='Comprehensive Program')
    
    # Chain multiple changes
    mod.set_efficacy('inj', 0.99) \\
       .set_duration_months('inj', 36) \\
       .set_method_mix('inj', 0.40, baseline_sim=baseline)
    
    intv = mod.build()

Multiple Interventions
----------------------

::

    # Two different interventions
    mod1 = fp.MethodIntervention(year=2010)
    mod1.set_efficacy('impl', 0.995)
    
    mod2 = fp.MethodIntervention(year=2015)
    mod2.set_duration_months('impl', 48)
    
    # Apply both
    sim = fp.Sim(
        pars=pars,
        interventions=[mod1.build(), mod2.build()]
    )
    sim.run()

Method Names
============

Use these short names:

============  ==========================================
Name          Description
============  ==========================================
``pill``      Oral contraceptive pills
``inj``       Injectables (DMPA, etc.)
``impl``      Implants (Jadelle, Implanon, etc.)
``iud``       Intrauterine devices
``cond``      Male condoms
``btl``       Female sterilization
``wdraw``     Withdrawal
``othtrad``   Other traditional methods
``othmod``    Other modern methods
============  ==========================================

Quick Tips
==========

✓ **Duration changes have large effects** - this is realistic! People accumulate on methods they stay on longer.

✓ **Chain methods** for cleaner code: ``mod.set_efficacy(...).set_duration_months(...)``

✓ **Method mix needs baseline** - always provide ``baseline_sim`` parameter

✓ **set_probability_of_use only works with RandomChoice** - for SimpleChoice/StandardChoice, use duration/method mix instead

✗ **Don't set efficacy > 1** - values must be between 0 and 1

✗ **Don't forget to call build()** - the modifier itself isn't an intervention

Functional Shortcut
===================

For single-method simple changes, use the functional shortcut::

    # Instead of:
    mod = fp.MethodIntervention(year=2025)
    mod.set_efficacy('pill', 0.95)
    intv = mod.build()
    
    # Use:
    intv = fp.make_update_methods(year=2025, method='pill', efficacy=0.95)

See Also
========

- :doc:`wrappers_guide` - Full documentation with examples
- :doc:`tutorials/T3_interventions_methods` - Interventions tutorial
- :doc:`tutorials/T5_new_method` - Adding new methods tutorial

