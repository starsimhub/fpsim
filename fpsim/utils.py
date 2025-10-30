'''
File for storing utilities and probability calculators needed to run FP model
'''

import numpy as np
import sciris as sc
import numba as nb
import scipy.stats as sps
from . import defaults as fpd
from . import version as fpv
from scipy import interpolate as si



# Specify all externally visible things this file defines
__all__ = ['DuplicateNameException', 'data2interp']


@nb.jit((nb.float32[:], ), cache=True, nopython=True)
def digitize_ages_1yr(ages):
    """
    Return the indices of the 1-year bins to which each value in ages array belongs.
    The bin index is used as an integer representation of the agent's age.
    """
    # Create age bins because ppl.age is a continous variable
    age_cutoffs = np.arange(0, fpd.max_age + 1)
    return np.digitize(ages, age_cutoffs) - 1


@nb.jit((nb.float64[:], nb.float64[:]), cache=True, nopython=True)
def digitize_ages(ages, age_group_lb):
    """
    This function returns the 0-based indices of the age bins passed in age_group_lb
    """
    return np.digitize(ages, age_group_lb) - 1  # returns 0-based indices of the group


@nb.njit((nb.float64[:], nb.float64, nb.float64), cache=True)
def numba_miscarriage_prob(miscarriage_rates, age, resolution):
    '''Run interpolation eval to check for probability of miscarriage here'''
    miscarriage_prob = miscarriage_rates[int(round(age*resolution))]
    return miscarriage_prob


def data2interp(data, ages, normalize=False):
    """ Convert unevenly spaced data into an even spline interpolation """
    model = si.interp1d(data[0], data[1])
    interp = model(ages)
    if normalize:
        interp = np.minimum(1, np.maximum(0, interp))
    return interp


def set_metadata(obj):
    ''' Set standard metadata for an object '''
    obj.created = sc.now()
    obj.version = fpv.__version__
    obj.git_info = sc.gitinfo(verbose=False)
    return


def piecewise_linear(x, x0, y0, m1, m2):
    '''
    Compute a two-part piecewise linear function at given x values.

    This function calculates the values of a piecewise linear function defined
    by two slopes (k1 and k2), a point of intersection (x0, y0), and an array
    of x values. The function returns an array of corresponding y values based
    on the piecewise linear function.

    Args:
    x (array-like)
        The array of x values at which the piecewise linear function is evaluated
    x0 (float)
        The x-coordinate of the point of intersection between the two linear segments (inflection point)
    y0 (float)
        The y-coordinate of the point of intersection between the two linear segments
    m1 (float)
        The slope of the first linear segment (for x < x0).
    m2 (float)
        The slope of the second linear segment (for x >= x0).

    Returns:
    y : ndarray
        An array of y values corresponding to the piecewise linear function
        evaluated at the input x values.

    **Examples**::
    >>> x_values = np.array([1, 2, 3, 4, 5])
    >>> y_values = piecewise_linear(x_values, 3, 2, 1, -1)
    '''
    return np.piecewise(x, [x < x0], [lambda x:m1*x + y0-m1*x0, lambda x:m2*x + y0-m2*x0])


def logistic_5p(x, a, b, c, d, e):
    '''
    A logistic function with 5 parameters (5p) that enables asymmetry

    Args:
    x (array-like)
       The array of x values at which the function is evaluated
    a (float):
      Minimum value (baseline) as x -> -infinity.
    d (float):
      Maximum value (saturation) as x -> infinity.
    b (float):
      Slope parameter
    c (float):
      Value of x at which the function reaches the midpoint between a and d.
    e (float):
      Exponent parameter controlling asymmetry.
        - If e = 1, the curve is symmetric.
        - If e > 1, the curve asymptotes toward "a" more quickly than it asymptotes toward "d."
        - If e < 1, the curve asymptotes toward "d" more quickly than it asymptotes toward "a."

    Returns:
    y : (ndarray)
        An array of y values corresponding to the piecewise linear function
        evaluated at the input x values.
    '''

    return d + ((a - d)/(1.0 + np.exp(b*(x-c)))**e)


def logistic_5p_dfun(x, a, b, c, d, e):
    '''
    Derivative of the 5 paraemter logistic function, same parameters
    '''
    return b*(a - d)*e*np.exp(b*(-c + x))*(1.0 + np.exp(b*(-c + x)))**(-1.0 - e)


def sigmoid_product(x, a1, b1, a2, b2):
    '''
    A product of two sigmoid functions. A monotonically increasing sigmoidal curve,
    followed by a monotonically decreasing sigmoidal curve.

    Current form produces  0 <= f(x) <= 1
    '''
    max_exp = 709
    x1 = np.clip(a1 - b1*x, -max_exp, max_exp)
    x2 = np.clip(a2 - b2*x, -max_exp, max_exp)
    return (1.0 / (1.0 + np.exp(x1))) * (1.0 / (1.0 + np.exp(x2)))


def gompertz(x, a, b, c):
    '''
    Compute the Gompertz function for a given set of parameters.
    This function is used for describing mortality and ageing-like processes.

    See:
    https://en.wikipedia.org/wiki/Gompertz_function

    The Gompertz function is defined as:
    f(x) = a * exp(-b * exp(-c * x))

    Parameters:
    x (array-like): The array of x values at which the function is evaluated
    a (float): The asymptote of the function as x approaches infinity.
    b (float): Displacement along the x-axis
    c (float): The growth rate

    Returns:
    y : (ndarray)
        An array of y values corresponding to the gomeprtz function
        evaluated at the input x values.
    """
    '''
    return a*np.exp(-b*np.exp(-c*x))


def gompertz_dfun(x, a, b, c):
    '''
    Compute the derivative of the Gompertz function with respect to x for a given set of parameters.

    The derivative of the Gompertz function is defined as:
    f'(x) = a * b * c * exp(-(b / exp(c * x)) - c * x)

    Parameters:
    x (array-like): The array of x values at which the function is evaluated
    a (float): The asymptote of the Gompertz function as x approaches infinity.
    b (float): Displacement along the x-axis
    c (float): The growth rate

    Returns:
    ndarray: An array of derivative values corresponding to the input x values.
    '''
    return a*b*c*np.exp(-(b/np.exp(c*x)) - c*x)


#% Exceptions

class DuplicateNameException(Exception):
    """
    Raised when either multiple instances of Module or State, or of any other type
    passed to ndict have duplicate names."""


    def __init__(self, obj):
        msg = f"A {type(obj)} with name `{obj.name}` has already been added."
        super().__init__(msg)
        return