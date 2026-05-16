'''
Benchmark how much longer empowerment-related functions take.
Education adds 7% to the runtime of a simulation.
'''

import sciris as sc
import fpsim as fp
import fpsim.education as fpemp


do_profile = 1
ms = fp.EmpoweredChoice(location='kenya')
emp = fp.Empowerment(location='kenya')
edu = fp.Education(location='kenya')
sim = fp.Sim(contraception_module=ms, empowerment_module=emp, education_module=edu)
sim.initialize()

ppl = sim.people
to_profile = sc.objdict(
    run                 =   sim.run,
    ppl_update    =  ppl.update,
    ppl_init      =  ppl.__init__,
    education     =  edu.update,
    edu_start     =  edu.start_education,      # 12% of update_education()
    edu_advance   =  edu.advance_education,    # 56% of update_education()
    edu_interrupt =  edu.interrupt_education,  # 19% of advance_education()
    edu_dropout   =  edu.dropout_education,    # 30% of advance_education() -- called twice
    edu_resume    =  edu.resume_education,     # 16% of update_education()
    edu_done      =  edu.graduate,             # 14% of update_education()
    empowerment   =  emp.update,
)


def run_with_empowerment():
    pars = fp.pars(location='kenya', n_agents=10e1, verbose=0)
    sim = fp.Sim(pars, contraception_module=ms, empowerment_module=emp, education_module=edu)
    sim.run()
    return sim


def run_without_empowerment():
    pars = fp.pars(location='kenya', n_agents=10e1, verbose=0)
    coefficients = sc.objdict(intercept=.1, age=2, parity=3)
    ms = fp.SimpleChoice(coefficients)
    sim = fp.Sim(pars, contraception_module=ms)
    sim.run()
    return sim


if __name__ == '__main__':
    selection = 'ppl_update'
    if do_profile:
        sc.profile(run_with_empowerment, to_profile[selection])
        sc.profile(run_without_empowerment, to_profile[selection])
    else:
        sc.tic()
        sim = run_with_empowerment()
        sc.toc()
