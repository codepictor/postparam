import numpy as np
import scipy as sp
import scipy.signal

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        b, K = self.true_params
        L = self._get_fixed_params()['L']
        R = self._get_fixed_params()['R']
        J = self._get_fixed_params()['J']

        A = np.array([
            [-b / J,  K / J],
            [-K / L, -R / L]
        ])
        B = np.array([
            [0, -1 / J],
            [1 / L, 0]
        ])
        C = np.array([
            [1, 0],
            [0, 1]
        ])
        D = np.array([
            [0, 0],
            [0, 0]
        ])
        sys = sp.signal.StateSpace(A, B, C, D)

        min_t = 0.0
        max_t = 25.0
        dt = 0.01
        tin = np.arange(min_t, max_t, step=dt)

        omega0 = 2 * np.pi
        V = 12 + 3 * np.cos(10 * omega0 * tin) + 2 * np.sin(2 * omega0 * tin)
        T = 0.05 + 0.05 * np.sin(omega0 * tin)
        T[:200] = 0

        tout, yout, xout = sp.signal.lsim(
            sys,                 # linear system
            np.array([V, T]).T,  # inputs
            tin,                 # time
            X0=None              # initial conditions
        )
        assert (np.diff(tout) > dt * 99999999 / 100000000).all()
        assert (np.diff(tout) < dt * 100000001 / 100000000).all()
        assert np.array_equal(xout, yout)

        return {
            'inputs': np.array([V, T]),
            'outputs': xout.T,
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
            'J': 0.01
        }

    @property
    def true_params(self):
        return np.array([0.1, 0.01])  # b, K

    @property
    def param_uncertainty(self):
        return np.array([0.5, 0.5])

    @property
    def system_name(self):
        return 'motor'

    @property
    def params_names(self):
        return ['b', 'K']

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

