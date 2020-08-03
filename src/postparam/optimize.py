"""Minimization of the objective function."""


import scipy as sp


def minimize(func, x0):
    """Minimize the objective function.

    Args:
        func (ObjectiveFunction): A function to minimize.
        x0 (numpy.ndarray): A starting point
            (typically, prior parameters of a dynamical system).

    Returns:
        out (numpy.ndarray): Found minimum.
    """
    if len(x0.shape) != 1:
        raise ValueError('A starting point must be one-dimensional array.')

    opt_res = sp.optimize.minimize(
        fun=lambda x: func.compute(x),
        x0=x0,
        method=None
    )
    # print(opt_res)
    return opt_res.x

