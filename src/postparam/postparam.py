"""Bayesian framework to infer parameters of a dynamical system.

This module contains all functions which should be used by a user.
It is not necessary to import other modules.

The typical baseline structures as follows:
a user collects data in time domain, transforms it to frequency domain
and gets posterior parameters of a dynamical system from prior ones.

"""

import numpy as np
import sympy

from . import objective_function
from . import data
from . import optimize


def prepare_time_data(inputs, outputs, dt):
    """Initialize the TimeData class based on raw data in time domain.

    Args:
        inputs (np.ndarray): Input data in time domain
            with shape (n_inputs, n_data_points).
        outputs (np.ndarray): Output data in time domain
            with shape (n_outputs, n_data_points).
        dt (double): Time step between data points.

    Returns:
        out (TimeData): Data in time domain
            which are stored in the appropriate class.
    """
    if inputs.shape[1] != outputs.shape[1]:
        raise ValueError('Number of points in inputs and outputs '
                         'must be equal.')
    if dt <= 0.0:
        raise ValueError('dt must be a positive number')

    return data.TimeData(
        inputs=inputs.copy(), outputs=outputs.copy(), dt=dt,
        input_std_devs=None, output_std_devs=None
    )


def prepare_freq_data(time_data, min_freq=None, max_freq=None):
    """Transform data from time domain to frequency domain.

    Args:
        time_data (TimeData): The object containing data in time domain.
        min_freq (double, optional): The left border of analyzing data
            in frequency domain. Defaults to None that is equivalent to 0.
        max_freq (double, optional): The right border of analyzing data
            in frequency domain. Defaults to None which means that
            all available frequencies will be used.

    Returns:
        freq_data (FreqData): Data after the DFT.
    """
    if min_freq < 0.0:
        raise ValueError('min_freq can not be negative.')
    if min_freq > max_freq:
        raise ValueError('min_freq must be less or equal to max_freq.')
    if time_data.input_std_devs is None or time_data.output_std_devs is None:
        raise ValueError('Measurement noise is not specified.')

    freq_data = data.FreqData(time_data)
    freq_data.trim(min_freq, max_freq)
    return freq_data


def compute_posterior_params(freq_data, admittance_matrix,
                             prior_params, prior_params_std):
    """Calculate posterior parameters using Bayesian inference.

    Args:
        freq_data (FreqData): Data after transformation
            from time domain to frequency domain.
        admittance_matrix (AdmittanceMatrix): An object representing
            an admittance matrix of a dynamical system.
        prior_params (numpy.ndarray): Prior parameters of a system.
        prior_params_std (numpy.ndarray): Prior uncertainties
            in system parameters.

    Returns:
        posterior_params (numpy.ndarray): Posterior parameters
            of a dynamical system calculated by employing the
            Bayesian inference.
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
    """Calculate outputs using the admittance matrix and inputs.

    Args:
        freqs (numpy.ndarray): Frequencies.
        freq_data_inputs (numpy.ndarray): Input data in frequency domain.
        admittance_matrix (AdmittanceMatrix): An object representing
            the admittance matrix of a dynamical system.
        dynsys_params (numpy.ndarray): Parameters of a dynamical system
            which should be substituted into the admittance matrix.

    Returns:
        predictions (numpy.ndarray): The admittance matrix
            multiplied by inputs for all frequencies.
    """
    if freq_data_inputs.shape[0] != admittance_matrix.data.shape[1]:
        raise ValueError('Number of arrays in output data should be equal to '
                         'number of rows that the admittance matrix has.')
    if freqs.shape[0] != freq_data_inputs.shape[1]:
        raise ValueError('Number of frequencies should be equal to '
                         'number of data points.')

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

