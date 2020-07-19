"""Bayesian framework for parameters identification of first order systems.

This module contains all functions which should be used by a user of
the framework. It is not necessary to import other modules.

The typical baseline structures as follows:
a user collects data in time domain, transforms it to frequency domain
by employing 'prepare_data' function and obtains posterior parameters
of an abstract dynamical system via 'compute_posterior_params'.
To use the latter function it is necessary to define your own class
representing the admittance matrix of your system.
The format of the admittance matrix is fixed.
See examples for more detail.

"""

import copy

import numpy as np
import sympy

from . import objective_function
from . import data
from . import optimize


def prepare_time_data(inputs, outputs, dt):
    """"""
    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError('Number of points in inputs and outputs '
                         'must be equal.')
    if dt <= 0.0:
        raise ValueError('dt must be a positive number')

    return data.TimeData(
        inputs=inputs, outputs=outputs, dt=dt,
        input_std_devs=None, output_std_devs=None
    )


def prepare_freq_data(time_data, snr=None, min_freq=None, max_freq=None):
    """Transform data from time domain to frequency domain.

    Args:
        time_data (TimeData): The object containing data in time domain.
        snr (double, optional): The value of SNR specifying noise
            which will be applied to data in time domain.
            If None, there will be no applying of any noise.
        min_freq (double, optional): The left border of analyzing data
            in frequency domain. Defaults to None that is equivalent to 0.
        max_freq (double, optional): The right border of analyzing data
            in frequency domain. Defaults to None which means that
            all frequencies will be used.

    Returns:
        freq_data (FreqData): Data after transformation from time domain
            to frequency domain.
    """
    if snr is not None and snr <= 0.0:
        raise ValueError('SNR should be positive.')
    if min_freq < 0.0:
        raise ValueError('min_freq can not be negative.')
    if min_freq > max_freq:
        raise ValueError('min_freq must be less or equal to max_freq.')

    time_data_copy = copy.deepcopy(time_data)

    if snr is not None:
        time_data_copy.apply_white_noise(snr)
    elif time_data.input_std_devs is None or time_data.output_std_devs is None:
        raise ValueError('Measurement noise is not specified.')

    freq_data = data.FreqData(time_data_copy)
    freq_data.trim(min_freq, max_freq)
    return freq_data


def compute_posterior_params(freq_data, admittance_matrix,
                             prior_params, prior_params_std):
    """Calculate posterior parameters employing Bayesian approach.

    Args:
        freq_data (FreqData): Data after transformation from time domain
            to frequency domain produced by the 'prepare_data' function.
        admittance_matrix (AdmittanceMatrix): User-defined class
            representing an admittance matrix of a dynamical system.
        prior_params (numpy.ndarray): Prior parameters of a system.
        prior_params_std (numpy.ndarray): Prior uncertainties in
            system parameters (see the 'perturb_params' function).

    Returns:
        posterior_params (numpy.ndarray): Posterior parameters
            of a dynamical system calculated by employing Bayesian
            approach and special optimization routine.
    """
    if (len(freq_data.outputs) != admittance_matrix.data.shape[0] or
            len(freq_data.inputs) != admittance_matrix.data.shape[1]):
        raise ValueError('Inconsistent shapes of data and admittance matrix.')
    if len(prior_params.shape) != 1 or len(prior_params_std.shape) != 1:
        raise ValueError('Prior parameters and deviations '
                         'must be one-dimensional numpy arrays.')
    if prior_params.shape != prior_params_std.shape:
        raise ValueError('Number of system parameters is not equal '
                         'to number of deviation fractions.')

    obj_func = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        admittance_matrix=admittance_matrix,
        prior_params=prior_params,
        prior_params_std=prior_params_std
    )

    posterior_params = optimize.minimize(func=obj_func, x0=prior_params)
    return posterior_params


def predict_outputs(freqs, freq_data_inputs, admittance_matrix, dynsys_params):
    """Calculate outputs based on admittance matrix and inputs."""
    n_inputs = admittance_matrix.data.shape[1]
    n_outputs = admittance_matrix.data.shape[0]
    n_freqs = len(freqs)
    n_params = len(dynsys_params)
    assert n_inputs == freq_data_inputs.shape[0]
    assert n_freqs == freq_data_inputs.shape[1]

    Y = admittance_matrix.data
    computed_Y = np.zeros((n_outputs, n_inputs, n_freqs), dtype=np.complex64)
    for row_idx in range(Y.shape[0]):
        for column_idx in range(Y.shape[1]):
            element_expr = Y[row_idx, :][column_idx]
            for param_idx in range(n_params):
                element_expr = element_expr.subs(
                    admittance_matrix.params[param_idx],
                    dynsys_params[param_idx]
                )
            computed_Y[row_idx][column_idx] = sympy.lambdify(
                args=admittance_matrix.omega,
                expr=element_expr,
                modules='numexpr'
            )(2.0 * np.pi * freqs)

    predictions = np.zeros((n_outputs, n_freqs), dtype=np.complex64)
    for freq_idx in range(n_freqs):
        predictions[:, freq_idx] = (
            computed_Y[:, :, freq_idx] @ freq_data_inputs[:, freq_idx]
        )
    return predictions

