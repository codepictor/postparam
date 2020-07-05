import time

import numpy as np

from circuit1 import dynamical_system
from postparam import postparam, objective_function
import utils


def debug(dynsys, snr, time_data, freq_data, prior_params, posterior_params):
    utils.plot_time_data(
        inputs=time_data.inputs,
        outputs=time_data.outputs,
        dynsys_name=dynsys.system_name,
        dt=time_data.dt
    )

    obj_func = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        admittance_matrix=dynsys.admittance_matrix,
        prior_params=prior_params,
        prior_params_std=dynsys.param_uncertainty
    )
    utils.plot_objective_function(
        obj_func=obj_func,
        true_params=dynsys.true_params,
        param_names=dynsys.params_names,
        dynsys_name=dynsys.system_name
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


def handle_one_snr(dynsys, time_data, snr):
    assert dynsys is not None
    assert time_data is not None
    assert 1.0 <= snr <= 100.0

    # for _ in range(50):
    freq_data = postparam.prepare_freq_data(
        time_data=time_data,
        snr=snr,
        remove_zero_freq=True,
        min_freq=dynsys.min_freq,
        max_freq=dynsys.max_freq
    )

    start_time = time.time()
    prior_params = dynsys.perturb_params()
    posterior_params = postparam.compute_posterior_params(
        freq_data=freq_data,
        admittance_matrix=dynsys.admittance_matrix,
        prior_params=prior_params,
        prior_params_std=dynsys.param_uncertainty
    )
    end_time = time.time()

    debug(
        dynsys, snr, time_data, freq_data,
        prior_params, posterior_params
    )

    return {
        'snr': snr,
        'prior_params': prior_params,
        'posterior_params': posterior_params,
        'optimization_time': end_time - start_time
    }


def main():
    dynsys = dynamical_system.DynamicalSystem()
    time_data = postparam.prepare_time_data(**dynsys.generate_time_data())

    snrs = 1.0 * np.arange(1, 21, 1)
    opt_info = []

    for snr in snrs:
        # print('SNR =', snr)
        opt_info.append(handle_one_snr(dynsys, time_data, snr))
        # print('prior_params:', opt_info[-1]['prior_params'])
        # print('posterior_params:', opt_info[-1]['posterior_params'])

    opt_time = np.array([snr_info['optimization_time'] for snr_info in opt_info])
    print('optimization time mean =', opt_time.mean(), '(seconds)')
    print('optimization time std  =', opt_time.std(), '(seconds)')

    utils.plot_params_convergence(
        dynsys=dynsys,
        snrs=snrs,
        prior_values=np.array([snr_info['prior_params'] for snr_info in opt_info]),
        posterior_values=np.array([snr_info['posterior_params'] for snr_info in opt_info])
    )


if __name__ == '__main__':
    main()

