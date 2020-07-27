"""Auxiliary functions to produce some plots."""

import os
import os.path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.titlesize'] = 44
matplotlib.rcParams['axes.labelsize'] = 30
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['lines.linewidth'] = 4
matplotlib.rcParams['lines.markersize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 30
matplotlib.rcParams['ytick.labelsize'] = 30
matplotlib.rcParams['legend.fontsize'] = 30
matplotlib.rcParams['legend.frameon'] = False
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 30
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['grid.linestyle'] = 'dashed'
plt.style.use(['bmh'])
plt.rc('text', usetex=True)


def plot_time_data(time_data, dynsys, plot_name):
    """Plot inputs and outputs in time domain."""
    assert time_data.inputs.shape[0] > 0
    assert time_data.outputs.shape[0] > 0
    assert time_data.dt > 0
    assert len(plot_name) > 0

    n_sensors = time_data.inputs.shape[0] + time_data.outputs.shape[0]
    t = np.array([i * time_data.dt for i in range(time_data.inputs.shape[1])])
    fig, axes = plt.subplots(
        n_sensors, 1,
        figsize=(25, 5 * n_sensors), sharex='all'
    )

    for input_idx in range(time_data.inputs.shape[0]):
        axes[input_idx].plot(
            t, time_data.inputs[input_idx]
        )
        axes[input_idx].set_ylabel(
            '$' + dynsys.inputs_names[input_idx] + ' (t)$'
        )

    for output_idx in range(time_data.outputs.shape[0]):
        axes[time_data.inputs.shape[0] + output_idx].plot(
            t, time_data.outputs[output_idx]
        )
        axes[time_data.inputs.shape[0] + output_idx].set_ylabel(
            '$' + dynsys.outputs_names[output_idx] + ' (t)$'
        )

    plt.xlabel('$t$ (seconds)')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name, plot_name + '.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    plt.close(fig)


def plot_freq_data(freq_data, dynsys, plot_name):
    """Plot inputs and outputs in frequency domain."""
    assert freq_data.inputs.shape[0] > 0
    assert freq_data.outputs.shape[0] > 0
    assert freq_data.freqs.shape[0] == freq_data.inputs.shape[1]
    assert freq_data.freqs.shape[0] == freq_data.outputs.shape[1]
    assert len(plot_name) > 0

    n_sensors = freq_data.inputs.shape[0] + freq_data.outputs.shape[0]
    fig, axes = plt.subplots(
        n_sensors, 1,
        figsize=(25, 5 * n_sensors), sharex='all'
    )

    for input_idx in range(freq_data.inputs.shape[0]):
        axes[input_idx].plot(
            freq_data.freqs, np.abs(freq_data.inputs[input_idx])
        )
        axes[input_idx].set_ylabel(
            '$|' + dynsys.inputs_names[input_idx] + ' (\\nu)|$'
        )

    for output_idx in range(freq_data.outputs.shape[0]):
        axes[freq_data.inputs.shape[0] + output_idx].plot(
            freq_data.freqs, np.abs(freq_data.outputs[output_idx])
        )
        axes[freq_data.inputs.shape[0] + output_idx].set_ylabel(
            '$|' + dynsys.outputs_names[output_idx] + ' (\\nu)|$'
        )

    plt.xlabel('$\\nu$ (Hz)')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name, plot_name + '.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    plt.close(fig)


def plot_objective_function(obj_func, dynsys):
    """Make 1-D plots of the objective function."""
    n_params = len(dynsys.true_params)
    n_points = 500
    fig, axes = plt.subplots(n_params, 1, figsize=(25, 10 * n_params))

    for param_idx in range(n_params):
        true_param = dynsys.true_params[param_idx]
        args = np.tile(dynsys.true_params, (n_points, 1))
        args[:, param_idx] = 1.0 * np.linspace(
            start=-true_param, stop=3*true_param, num=n_points
        )

        param_name = dynsys.params_names[param_idx]
        ax = axes[param_idx] if n_params > 1 else axes
        ax.plot(args[:, param_idx], obj_func.compute(args))
        ax.axvline(true_param, alpha=0.75, color='red')
        ax.set_xlabel('$' + param_name + '$')
        ax.set_ylabel('$f(' + param_name + ')$')

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name,
            'objective_function.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    plt.close(fig)


def plot_measurements_and_predictions_abs(freqs, measurements, predictions,
                                          dynsys, plot_name):
    """Plot absolute values of measured and predicted data."""
    assert len(freqs.shape) == 1
    assert len(measurements.shape) == len(predictions.shape) == 2
    assert measurements.shape[0] == predictions.shape[0] == len(dynsys.outputs_names)
    assert len(freqs) == measurements.shape[1] == predictions.shape[1]
    assert len(plot_name) > 0

    n_outputs = measurements.shape[0]
    fig, axes = plt.subplots(
        n_outputs, 1,
        figsize=(25, 10 * n_outputs), sharex='all'
    )

    for output_idx in range(n_outputs):
        ax = axes[output_idx] if n_outputs > 1 else axes
        ax.set_yscale('log')
        ax.plot(
            freqs, np.abs(measurements[output_idx]),
            label='measured', color='black'
        )
        ax.plot(
            freqs, np.abs(predictions[output_idx]),
            label='predicted', color='blue'
        )
        ax.set_ylabel(
            '$\\log (|' + dynsys.outputs_names[output_idx] + ' (\\nu)|)$'
        )
        ax.legend(loc='upper right')

    plt.xlabel('$\\nu$ (Hz)')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name, plot_name + '.pdf'
        ),
        dpi=180, format='pdf'
    )
    plt.close('all')


def plot_params_convergence(snrs, prior_values, posterior_values, dynsys):
    """Plot convergence of parameters for different SNR."""
    assert len(snrs.shape) == 1
    assert len(prior_values.shape) == len(posterior_values.shape) == 3
    assert len(snrs) == prior_values.shape[0] == posterior_values.shape[0]
    assert prior_values.shape[1] == posterior_values.shape[1]
    assert prior_values.shape[1] == len(dynsys.true_params)
    assert prior_values.shape[2] == posterior_values.shape[2]

    n_params = len(dynsys.true_params)
    fig, axes = plt.subplots(
        n_params, 1,
        figsize=(25, 10 * n_params), sharex='all'
    )

    for param_idx in range(n_params):
        ax = axes[param_idx] if n_params > 1 else axes
        _plot_param_convergence_prior(
            ax, snrs, prior_values=prior_values[:, param_idx, :]
        )
        _plot_param_convergence_posterior(
            ax, snrs, posterior_values=posterior_values[:, param_idx, :]
        )
        _plot_param_convergence_true(
            ax, snrs, true_value=dynsys.true_params[param_idx]
        )

        ax.set_ylim((
            0.5 * dynsys.true_params[param_idx],
            1.5 * dynsys.true_params[param_idx]
        ))
        ax.set_xlabel('SNR')
        ax.set_ylabel('$' + dynsys.params_names[param_idx] + '$')
        ax.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name, 'params_convergences.pdf'
        ),
        dpi=180, format='pdf'
    )
    plt.close(fig)


def _plot_param_convergence_prior(ax, snrs, prior_values):
    # Fill [prior_mean - prior_std; prior_mean + prior_std]
    assert len(prior_values.shape) == 2
    assert len(snrs) == prior_values.shape[0]

    mean = prior_values.mean(axis=1)
    std = prior_values.std(axis=1)
    ax.plot(
        snrs, mean,
        label='prior', linewidth=4, marker='o', color='b'
    )
    ax.fill_between(
        snrs, mean - std, mean + std,
        color='b', alpha=0.2
    )


def _plot_param_convergence_posterior(ax, snrs, posterior_values):
    # Fill [posterior_mean - posterior_std; posterior_mean + posterior_std]
    assert len(posterior_values.shape) == 2
    assert len(snrs) == posterior_values.shape[0]

    mean = posterior_values.mean(axis=1)
    std = posterior_values.std(axis=1)
    ax.plot(
        snrs, mean,
        label='posterior', linewidth=4, marker='o', color='g'
    )
    ax.fill_between(
        snrs, mean - std, mean + std,
        color='g', alpha=0.3
    )


def _plot_param_convergence_true(ax, snrs, true_value):
    # Draw the red line corresponding to true parameter
    ax.plot(
        snrs, np.repeat(true_value, len(snrs)),
        label='true', linewidth=4, linestyle='dashed', color='r'
    )

