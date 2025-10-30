"""
Test that the current version of FPsim exactly matches
the baseline results. To update baseline, run ./update_baseline.
"""


import numpy as np
import sciris as sc
import fpsim as fp
import starsim as ss

do_plot = 0
do_save = 0
baseline_filename  = sc.thisdir(__file__, 'baseline.json')
benchmark_filename = sc.thisdir(__file__, 'benchmark.json')


def make_exp(n_agents=10000, seed=1, do_run=False, do_plot=False):
    '''
    Define a default simulation for testing the baseline.

    Note: due to these being rare events, some sims will have zero maternal or
    infant mortality. If this happens it's OK to change the default seed. Run with
    a larger population size (e.g. 10k) to see what the "expected" value is, then
    find a seed for a small pop (e.g. 1k) which produces results closest to the
    large-pop values.
    '''
    pars = dict(n_agents=n_agents, seed=seed, verbose=0, test=True)
    exp = fp.Experiment(pars=pars)

    if do_run or do_plot:
        exp.run()

    if do_plot:
        exp.plot()

    return exp


def save_baseline():
    '''
    Refresh the baseline results. This function is not called during standard testing,
    but instead is called by the update_baseline script.
    '''
    print('Updating baseline values...')
    exp = make_exp(do_run=True)
    exp.to_json(filename=baseline_filename)
    print('Done.')
    return


def test_baseline():
    ''' Compare the current default sim against the saved baseline '''
    sc.heading('Testing baseline...')

    # Load existing baseline
    current = sc.loadjson(baseline_filename)

    # Calculate new baseline
    exp = make_exp(do_run=True)
    new = exp.summarize()

    # Compute the comparison
    fp.diff_summaries(current, new, die=True)

    return new


def test_benchmark(do_save=do_save, repeats=1):
    ''' Compare benchmark performance '''

    sc.heading('Running benchmark...')
    previous = sc.loadjson(benchmark_filename)

    t_inits = []
    t_runs  = []
    t_posts  = []

    def normalize_performance():
        ''' Normalize performance across CPUs -- simple Numpy calculation '''
        t_bls = []
        bl_repeats = 3
        n_outer = 10
        n_inner = 1e6
        for r in range(bl_repeats):
            t0 = sc.tic()
            for i in range(n_outer):
                a = np.random.random(int(n_inner))
                b = np.random.random(int(n_inner))
                a*b
            t_bl = sc.toc(t0, output=True)
            t_bls.append(t_bl)
        t_bl = min(t_bls)
        reference = 0.112 # Benchmarked on an Intel i9-8950HK CPU @ 2.90GHz
        ratio = reference/t_bl
        return ratio


    # Test CPU performance before the run
    r1 = normalize_performance()

    # Do the actual benchmarking
    for r in range(repeats):

        # Create the sim
        exp = make_exp()

        # Time running
        t0 = sc.tic()
        exp.run_model()
        t_run = sc.toc(t0, output=True)

        # Time initialization
        t0 = sc.tic()
        t_init = sc.toc(t0, output=True)

        # Time postprocessing
        t0 = sc.tic()
        exp.post_process_results()
        t_post = sc.toc(t0, output=True)

        # Store results
        t_inits.append(t_init)
        t_runs.append(t_run)
        t_posts.append(t_post)

    # Test CPU performance after the run
    r2 = normalize_performance()
    ratio = (r1+r2)/2
    t_init = min(t_inits)*ratio
    t_run  = min(t_runs)*ratio
    t_post = min(t_posts)*ratio

    # Construct json
    n_decimals = 3
    json = {'time': {
                'initialize':  round(t_init, n_decimals),
                'run':         round(t_run,  n_decimals),
                'postprocess': round(t_post, n_decimals),
                'total':       round(t_init+t_run+t_post, n_decimals)
                },
            'parameters': {
                'n':          exp.sim.pars['n_agents'],
                'start_year': exp.sim.pars['start'],
                'end_year':   exp.sim.pars['stop'],
                'timestep':   exp.sim.pars['dt'],
                },
            'cpu_performance': ratio,
            }

    def printjson(json):
        ''' Print more nicely '''
        print(sc.jsonify(json, tostring=True, indent=2))

    print('Previous benchmark:')
    printjson(previous)

    print('\nNew benchmark:')
    printjson(json)

    if do_save:
        sc.savejson(filename=benchmark_filename, obj=json, indent=2)

    print('Done.')

    return json



if __name__ == '__main__':

    # Start timing and optionally enable interactive plotting
    T = sc.tic()

    json = test_benchmark(do_save=do_save, repeats=1) # Run this first so benchmarking is available even if results are different
    new  = test_baseline()
    exp = make_exp(do_plot=True)

    print('\n'*2)
    sc.toc(T)
    print('Done.')
