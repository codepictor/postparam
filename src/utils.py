"""Auxiliary functions to make some plots."""

import os
import os.path

import numpy as np
import matplotlib.pyplot as plt


def plot_time_data(inputs, outputs, dt, dynsys_name):
    """"""

    _plot_time_data(inputs, dt, dynsys_name, plot_name='time_inputs')
    _plot_time_data(outputs, dt, dynsys_name, plot_name='time_outputs')


def _plot_time_data(data, dt, dynsys_name, plot_name):
    assert len(data.shape) == 2
    assert dt > 0
    assert len(plot_name) > 0

    n_sensors = data.shape[0]
    fig, axes = plt.subplots(
        n_sensors, 1,
        figsize=(data.shape[1] // 10, 4 * n_sensors)
    )

    for sensor_idx in range(data.shape[0]):
        _plot_time_data_array(
            ax=axes[sensor_idx] if n_sensors > 1 else axes,
            time_data_array=data[sensor_idx],
            sensor_name=('sensor' + str(sensor_idx))
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys_name, plot_name + '.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    plt.close(fig)


def _plot_time_data_array(ax, time_data_array, sensor_name):
    assert len(time_data_array.shape) == 1
    assert len(sensor_name) > 0

    ax.plot(time_data_array)
    ax.set_title(sensor_name)


def plot_objective_function(obj_func, true_params, param_names, dynsys_name):
    """Make 1-D plots of the objective function."""
    n_params = len(true_params)
    n_points = 500
    fig, axes = plt.subplots(n_params, 1, figsize=(16, 6 * n_params))

    for param_idx in range(n_params):
        true_param = true_params[param_idx]
        args = np.tile(true_params, (n_points, 1))
        args[:, param_idx] = 1.0 * np.linspace(
            start=-true_param, stop=3*true_param, num=n_points
        )
        obj_func_values = obj_func.compute(args)
        param_name = (
            param_names[param_idx] if param_names is not None
            else 'param' + str(param_idx)
        )

        ax = axes[param_idx] if len(true_params) > 1 else axes
        ax.set_title('Vary $' + param_name + '$', fontsize=20)
        ax.plot(args[:, param_idx], obj_func_values)
        ax.axvline(true_param, alpha=0.75, color='red')
        ax.set_xlabel('$' + param_name + '$')
        ax.set_ylabel('$f(' + param_name + ')$')

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys_name, 'objective_function.pdf'
        ),
        dpi=180,
        format='pdf'
    )
    plt.close(fig)


def plot_measurements_and_predictions(freqs, measurements, predictions,
                                      out_file_name, yscale=None, yticks=None,
                                      xlabel=None, ylabel=None):
    """Plot measured and predicted data."""
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(24, 8))

    plt.plot(freqs, measurements, label='Measured', color='black')
    plt.plot(freqs, predictions, label='Predicted', color='b')

    if yscale is not None:
        plt.yscale(yscale)
    plt.tick_params(
        axis='both', labelsize=60, direction='in', length=12, width=3, pad=12
    )
    plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=60)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=60)
    plt.grid(alpha=0.75)
    plt.legend(loc='upper left', prop={'size': 50}, frameon=False, ncol=1)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', 'predictions_output_data',
            out_file_name + '.pdf'
        ),
        dpi=180, format='pdf'
    )
    plt.close('all')


def plot_params_convergence(dynsys, snrs, prior_values, posterior_values):
    """Plot convergence of parameters for different SNR."""
    assert len(snrs) == len(posterior_values) == len(prior_values)
    assert prior_values.shape[1] == posterior_values.shape[1]

    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    n_params = len(dynsys.true_params)
    fig, axes = plt.subplots(n_params, 1, figsize=(24, 12 * n_params))

    for param_idx in range(n_params):
        _plot_param_convergence(
            ax=axes[param_idx] if n_params > 1 else axes,
            snrs=snrs,
            prior_values=prior_values[:, param_idx],
            posterior_values=posterior_values[:, param_idx],
            true_value=dynsys.true_params[param_idx],
            param_name=dynsys.params_names[param_idx]
        )

    fig.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', dynsys.system_name, 'params_convergences.pdf'
        ),
        dpi=180, format='pdf'
    )
    plt.close(fig)


def _plot_param_convergence(ax, snrs,
                            prior_values, posterior_values,
                            true_value, param_name):
    assert len(snrs) == len(prior_values) == len(posterior_values)
    # assert snrs == [1, 2, ..., 21]

    ax.plot(
        snrs, prior_values,
        label='prior', linewidth=4, marker='o', color='b'
    )
    ax.plot(
        snrs, posterior_values,
        label='posterior', linewidth=4, marker='o', color='g'
    )
    ax.plot(
        snrs, [true_value for _ in range(len(snrs))],
        label='true', linewidth=4, linestyle='dashed', color='r'
    )

    ax.grid(alpha=0.75)
    ax.tick_params(
        axis='both', labelsize=60, direction='in',
        length=12, width=3, pad=12
    )
    n_ticks = 5
    y_min = 0.0
    y_max = 2.0 * true_value
    step = (y_max - y_min) / (n_ticks - 1)
    ax.set_yticks(np.arange(y_min, y_max + step, step))
    ax.set_ylim((y_min, y_max))
    ax.set_xticks(range(0, len(snrs) + 1, 5))

    ax.set_xlabel('SNR', fontsize=60)
    ax.set_ylabel('$' + param_name + '$', labelpad=20, fontsize=60)
    ax.legend(loc='upper right', prop={'size': 60}, frameon=True, ncol=3)

