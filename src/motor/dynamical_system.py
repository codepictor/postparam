import numpy as np
import scipy as sp
import scipy.signal

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        b = self.true_params[0]
        L = self._get_fixed_params()['L']
        R = self._get_fixed_params()['R']
        J = self._get_fixed_params()['J']
        K = self._get_fixed_params()['K']

        A = np.array([
            [-b / J,  K / J],
            [-K / L, -R / L]
        ])
        B = np.array([
            [0],
            [1 / L]
        ])
        C = np.array([
            [1, 0]
        ])
        D = np.array([
            [0]
        ])
        sys = sp.signal.StateSpace(A, B, C, D)

        min_t = 0.0
        max_t = 10.0
        dt = 0.01
        tin = np.arange(min_t, max_t, step=dt)

        omega0 = 2 * np.pi * 10
        V0 = 2
        V = 12 + V0 * np.cos(omega0 * tin)
        # V = 12 * np.ones(len(tin))

        tout, yout, xout = sp.signal.lsim(sys, V, tin, X0=None)
        assert (np.diff(tout) > dt * 99999999 / 100000000).all()
        assert (np.diff(tout) < dt * 100000001 / 100000000).all()
        assert np.array_equal(xout[:, 0], yout)

        return {
            'inputs': np.array([V]),
            'outputs': np.array([yout]),
            'dt': dt
        }

    def perturb_true_params(self):
        perturbations = np.random.uniform(
            low=-self.param_uncertainty,
            high=self.param_uncertainty
        )
        perturbed_params = (perturbations + 1.0) * self.true_params
        return perturbed_params

    def _get_fixed_params(self):
        return {
            'L': 0.5,
            'R': 1.0,
            'J': 0.01,
            'K': 0.01
        }

    @property
    def true_params(self):
        return np.array([0.1])

    @property
    def param_uncertainty(self):
        return np.array([0.5])

    @property
    def system_name(self):
        return 'motor'

    @property
    def params_names(self):
        return ['b']

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 20.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix(
            **self._get_fixed_params()
        )

