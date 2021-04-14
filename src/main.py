import time

import numpy as np

from generator import dynamical_system
from postparam import postparam, objective_function
import utils


def debug(dynsys, snr, pure_time_data, noisy_time_data, freq_data,
          prior_params, posterior_params):
    utils.plot_time_data(pure_time_data, dynsys, plot_name='pure_time_data')
    utils.plot_time_data(noisy_time_data, dynsys, plot_name='noisy_time_data')
    utils.plot_freq_data(freq_data, dynsys, plot_name='freq_data_amplitudes')

    obj_func = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        admittance_matrix=dynsys.admittance_matrix,
        prior_params=prior_params,
        prior_params_std=dynsys.param_uncertainty
    )
    utils.plot_objective_function(
        obj_func, dynsys,
        prior_params, posterior_params
    )

    print('=============================================================')
    print('SNR =', snr)
    print('PRIOR PARAMETERS    :', prior_params)
    print('PRIOR PARAMETERS STD:', dynsys.param_uncertainty)
    print('POSTERIOR PARAMETERS:', posterior_params)
    print('TRUE PARAMETERS     :', dynsys.true_params)
    print('f(PRIOR PARAMETERS)     =', obj_func.compute(prior_params))
    print('f(POSTERIOR PARAMETERS) =', obj_func.compute(posterior_params))
    print('f(TRUE PARAMETERS)      =', obj_func.compute(dynsys.true_params))
    print('=============================================================')


def handle_one_snr(dynsys, pure_time_data, snr):
    assert dynsys is not None
    assert pure_time_data is not None
    assert 1.0 <= snr <= 100.0

    n_runs = 1
    prior_params = np.zeros((len(dynsys.true_params), n_runs))
    posterior_params = np.zeros((len(dynsys.true_params), n_runs))
    opt_time = np.zeros(n_runs)

    for i in range(n_runs):
        noisy_time_data = postparam.prepare_time_data(
            inputs=pure_time_data.inputs.copy(),
            outputs=pure_time_data.outputs.copy(),
            dt=pure_time_data.dt
        ).apply_white_noise(snr)

        freq_data = postparam.prepare_freq_data(
            time_data=noisy_time_data,
            min_freq=dynsys.min_freq,
            max_freq=dynsys.max_freq
        )

        prior_params[:, i] = dynsys.perturb_true_params()

        start_time = time.time()
        posterior_params[:, i] = postparam.compute_posterior_params(
            freq_data=freq_data,
            admittance_matrix=dynsys.admittance_matrix,
            prior_params=prior_params[:, i],
            prior_params_std=dynsys.param_uncertainty
        )
        end_time = time.time()
        opt_time[i] = end_time - start_time

        debug(
            dynsys, snr, pure_time_data, noisy_time_data, freq_data,
            prior_params[:, i], posterior_params[:, i]
        )

    return {
        'snr': snr,
        'prior_params': prior_params,
        'posterior_params': posterior_params,
        'elapsed_time': opt_time
    }


def main():
    dynsys = dynamical_system.DynamicalSystem()
    time_data = postparam.prepare_time_data(**dynsys.generate_time_data())
    snrs = 1.0 * np.arange(1, 21, 1)
    opt_info = []

    for snr in snrs:
        opt_info.append(handle_one_snr(dynsys, time_data, snr))

    utils.plot_params_convergence(
        dynsys=dynsys,
        snrs=snrs,
        prior_values=np.array([
            snr_info['prior_params'] for snr_info in opt_info
        ]),
        posterior_values=np.array([
            snr_info['posterior_params'] for snr_info in opt_info
        ])
    )
    utils.plot_optimization_time(
        snrs=snrs,
        optimization_time=np.array([
            snr_info['elapsed_time'] for snr_info in opt_info
        ]),
        dynsys=dynsys
    )


if __name__ == '__main__':
    main()

