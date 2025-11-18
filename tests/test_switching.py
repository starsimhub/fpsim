"""
Test switching matrix
"""

import fpsim as fp
import sciris as sc
import starsim as ss


par_kwargs = dict(test=True)
parallel = 1  # Whether to run in serial (for debugging)


def test_create():
    sc.heading('Test creating a switching matrix')
    methods = fp.make_methods()
    switch = fp.Switching(methods, location='kenya')
    new_matrix = switch.extend_matrix(new_name='inj2')
    assert len(switch.matrix[0]['25-35'][8]) == len(new_matrix[0]['25-35'][8]) - 1, "Switching matrix was not extended correctly"
    sc.printgreen('✓ (successfully created and extended switching matrix)')
    return switch


def test_modify():
    sc.heading('Test modifying a switching matrix')

    methods = fp.make_methods()
    sw = fp.Switching(methods, location='kenya')
    sw.set_entry('pill', 'iud', 0.4, postpartum=0)
    assert sw.get_entry('pill', 'iud', postpartum=0, age_grp='20-25') == 0.4, "Switching matrix entry was not set"
    return sw


def test_copy():
    sc.heading('Test copying rows/columns of switching matrix')

    methods = fp.make_methods()
    sw = fp.Switching(methods, location='kenya')
    sw.copy_from_method_column('iud', 'impl', postpartum=0, age_grp='20-25')
    sc.printgreen('✓ (successfully copied switching matrix row)')
    return sw


def test_add_method():
    sc.heading('Test adding a method to switching matrix')

    # Make methods, create new method
    methods = fp.make_methods()
    new_inj = sc.dcp(methods['inj'])
    new_inj.name = 'inj2'
    new_inj.efficacy = 0.99

    # Create switching matrix and add new method
    intv = fp.add_method(year=2002, method=new_inj, copy_from='inj')
    sim = fp.Sim(pars=par_kwargs, interventions=[intv])
    sim.run()
    return


if __name__ == '__main__':

    switch = test_create()
    sw = test_modify()
    sw = test_copy()
    test_add_method()

    print('Done.')

