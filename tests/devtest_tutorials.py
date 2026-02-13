"""
Test running tutorials
"""


def run_t1():
    import fpsim as fp

    sim = fp.Sim(location=location)
    sim.run()
    fig = sim.plot()
    pars = dict(
            n_agents   = 10_000,
            location   = 'kenya',
            start_year = 2000,
            end_year   = 2020,
        )

    sim = fp.Sim(pars)
    sim.run()
    sim.results.fp['pregnancies']
    sim.list_available_results()
    sim = fp.Sim(n_agents=10e3, location='kenya', start_year=2000, end_year=2020)
    sim.run()

    sim = fp.Sim(pars, n_agents=100) # Use parameters defined above, but overrides the value `n_agents` in pars, and uses instead 100 agents, not 10,000
    sim.run()

    sim.plot() # the default
    sim.plot('cpr')


def run_t2():
    # Import FPsim and define the baseline parameters
    import fpsim as fp
    location = 'kenya'
    pars = dict(location=location, n_agents=500, start_year=1980, end_year=2020, seed=1)

    # Define the contraceptive choice module and the education module.
    choice = fp.StandardChoice(location=location)
    edu = fp.Education(location=location)

    # Make and run sim
    s = fp.Sim(pars, contraception_module=choice, education_module=edu)
    s.run()

    method_choice = fp.SimpleChoice(location=location)
    sim = fp.Sim(pars=pars, contraception_module=method_choice, analyzers=fp.lifeof_recorder())
    sim.run()

    _ = sim.plot(to_plot='cpr');
    _ = sim.analyzers[0].plot(index=1); # plot the life events of one woman

    return


def run_t3():

    import fpsim as fp
    import sciris as sc
    import starsim as ss

    pars = dict(
        n_agents   = 1_000,
        location   = 'kenya',
        start_year = 2000,
        end_year   = 2020,
        exposure_factor = 1.0  # Overall scale factor on probability of becoming pregnant
    )
    method_choice = fp.RandomChoice()

    s1 = fp.Sim(pars=pars, contraception_module=method_choice, label="Baseline")
    Methods = fp.make_methods()
    for method in Methods.values(): print(f"{method.idx}: {method.label}")
    change_efficacy_intervention = fp.update_methods(eff={"Injectables": 0.99}, year=2010)  # new efficacy starts in 2010

    s2 = fp.Sim(pars=pars, contraception_module=method_choice,
                     interventions=change_efficacy_intervention,
                     label="More effective Injectables")
    # The baseline duration for Injectables is a lognormal with parameter par1=2, and par2=3
    change_duration_intervention = fp.update_methods(dur_use={'Injectables': dict(dist='lognormal', par1=3, par2=0.2)}, year=2010)

    # Define a simulaiton for this intervention called s3
    s3 = fp.Sim(pars=pars, contraception_module=method_choice,
                     interventions=change_duration_intervention,
                     label="Longer time on Injectables")
    # The values in method_mix should add up to 1, but if they don't, the intervention update_methods() will autamotailly normalize them to add up to 1.
    change_mix = fp.update_methods(method_mix=[0.25, 0.05, 0.05, 0.0, 0.05, 0.3, 0.1, 0.1, 0.0], year=2010.0)

    # Define a simulation for this intervention called s4
    s4 = fp.Sim(pars=pars, contraception_module=method_choice,
                interventions=change_mix,
                label='Different mix')
    simlist = sc.autolist([s1, s2, s3, s4])
    msim = ss.MultiSim(sims=simlist)
    msim.run(parallel=False)
    msim.plot(key='cpr')


def run_t4():
    # T4
    import fpsim as fp      # Main FPsim module
    import sciris as sc     # For utilities
    import starsim as ss    # For running multiple sims in parallel
    import matplotlib.pyplot as plt  # For plotting

    location = 'kenya'
    pars = dict(
        n_agents   = 1_000,
        location   = location,
        start_year = 2000,
        end_year   = 2012,
        exposure_factor = 1.0  # Overall scale factor on probability of becoming pregnant
    )

    from fpsim.defaults import fpmod_states
    for i, state in enumerate(fpmod_states): print(f"{i}: {state.name}")

    def select_undereducated(sim):
        """ Select women who have a goal to achieve at least 1 year of education """
        is_eligible = ((sim.people.female) &
                       (sim.people.alive)     &
                       (sim.people.edu.objective > 0))
        return is_eligible

    edu = fp.Education(location=location)
    s0 = fp.Sim(pars=pars, education_module=edu, label='Baseline')

    change_education = fp.change_people_state(
                                'edu.attainment',
                                eligibility=select_undereducated,
                                years=2010.0,
                                new_val=15,  # Give all selected women 15 years of education
                            )
    edu = fp.Education(location=location)
    s1 = fp.Sim(pars=pars,
                education_module=edu,
                interventions=change_education,
                label='Increased education')

    msim = ss.MultiSim(sims=[s0, s1])
    msim.run(parallel=False)
    s0, s1 = msim.sims

    plt.plot(s0.results.timevec, s0.results.edu.mean_attainment, label=s0.label)
    plt.plot(s1.results.timevec, s1.results.edu.mean_attainment, label=s1.label)
    plt.ylabel('Average years of education among women')
    plt.xlabel('Year')
    plt.legend()


def run_t5():
    import fpsim as fp
    import sciris as sc
    import starsim as ss
    my_methods = fp.make_methods()
    new_method = fp.Method(name='new', efficacy=0.96,  modern=True,  dur_use=15, label='New method')
    my_methods += new_method
    # Note: if we do not define this method mix, the contraception module will use 1/(number of methods) for every method.
    method_choice = fp.RandomChoice(methods=my_methods, pars={'method_mix': [0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0.5]})

    pars = dict(
        n_agents   = 10_000,
        start_year = 2000,
        end_year   = 2012,
        exposure_factor = 1.0  # Overall scale factor on probability of becoming pregnant
    )

    s1 = fp.Sim(pars=pars, label='Baseline')
    s2 = fp.Sim(pars=pars, contraception_module=method_choice, label='New Method')
    simlist = sc.autolist([s1, s2])
    msim = ss.MultiSim(sims=simlist)
    msim.run(parallel=False)

    msim.plot(key='cpr')


def run_t6():
    import fpsim as fp
    from fpsim import plotting as plt
    import numpy as np

    country = 'kenya'
    pars = fp.all_pars(location=country)  # For default pars

    # Initial free parameters for calibration
    pars['exposure_factor'] = 1

    # Postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    pars['spacing_pref']['preference'][3:6] = 1  # Spacing of 9-15 months
    pars['spacing_pref']['preference'][6:9] = 1  # Spacing of 18-24 months
    pars['spacing_pref']['preference'][9:] =  1  # Spacing of 27-36 months

    # Only other simulation free parameters are age-based exposure and parity-based exposure (which you can adjust manually in {country}.py) as well as primary_infertility (set to 0.05 by default)

    # Adjust contraceptive choice parameters
    cm_pars = dict(
        prob_use_year=2020,  # Time trend intercept
        prob_use_trend_par=0.06,   # Time trend parameter
        method_weights=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').
    )
    method_choice = fp.SimpleChoice(pars=cm_pars, location=country)     # The contraceptive choice module used (see methods.py for more documentation). We can select RandomChoice, SimpleChoice, or StandardChoice (StandardChoice is selected by default).

    # Run the sim
    sim = fp.Sim(pars=pars, contraception_module=method_choice)
    sim.run()

    # # Plot sim
    # sim.plot()
    #
    # # Plotting class function which plots the primary calibration targets (method mix, method use, cpr, total fertility rate, birth spacing, age at first birth, and age-specific fertility rate)
    # plt.plot_calib(sim)

    # Initial free parameters for calibration
    pars['exposure_factor'] = 2

    # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    pars['spacing_pref']['preference'][:4] = .4  # Spacing of 0-12 months
    pars['spacing_pref']['preference'][4:8] = .2  # Spacing of 12-24 months
    pars['spacing_pref']['preference'][8:16] = 2  # Spacing of 24-48 months
    pars['spacing_pref']['preference'][16:] = 1  # Spacing of >48 months

    # Re-run the sim
    method_choice = fp.SimpleChoice(pars=cm_pars, location=country)
    sim = fp.Sim(pars=pars, contraception_module=method_choice)
    sim.run()
    # plt.plot_calib(sim)

    # Initial free parameters for calibration
    pars['exposure_factor'] = 2.5

    # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'. Pulls values from {location}/data/birth_spacing_pref.csv by default
    # Set all to 1 to reset. Option to use 'optimize-space-prefs.py' script in this directory to determine values
    pars['spacing_pref']['preference'][:4] = .4  # Spacing of 0-12 months
    pars['spacing_pref']['preference'][4:8] = .2  # Spacing of 12-24 months
    pars['spacing_pref']['preference'][8:16] = 2  # Spacing of 24-48 months
    pars['spacing_pref']['preference'][16:] = 1  # Spacing of >48 months

    # Adjust contraceptive choice parameters
    cm_pars = dict(
        prob_use_year=2020,  # Time trend intercept
        prob_use_trend_par=0.06,   # Time trend parameter
        force_choose=False,        # Whether to force non-users to choose a method ('False' by default)
        method_weights=np.array([.5, .5, 1, .7, 1, 1, 1.3, .8, 3])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').
    )
    method_choice = fp.SimpleChoice(pars=cm_pars, location=country)

    # Re-run the sim
    sim = fp.Sim(pars=pars, contraception_module=method_choice)
    # sim.run()
    # plt.plot_calib(sim)

    pars = dict(location = country,
                n_agents = 1000,  # Population size; set very small here only for the purpose of runtime
                end_year = 2020  # 1960 - 2020 is the normal date range
    )

    # Free parameters for calibration
    freepars = dict(
            exposure_factor = [1, 0.5, 2],       # [best, low, high]
    )
    calibration = fp.Calibration(pars, calib_pars=freepars, n_trials=2)
    calibration.calibrate()
    calibration.summarize()

    # fig = calibration.plot_trend()
    # fig = calibration.plot_best()
    return


if __name__ == '__main__':

    # run_t1()
    # run_t2()
    # run_t3()
    # run_t4()
    # run_t5()
    run_t6()

    print('Done.')
