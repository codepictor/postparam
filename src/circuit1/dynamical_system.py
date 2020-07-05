import numpy as np

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        """Simulate data in time domain."""
        epsilon0 = 3
        omega0 = 2 * np.pi * 4  # 4 Hz
        tau = self.true_params[0]

        min_t = 0.0
        max_t = 100.0
        dt = 0.05

        t = np.arange(min_t, max_t, step=dt)
        inputs = np.array([
            epsilon0 * np.cos(omega0 * t)
        ])
        outputs = np.array([
            (epsilon0 / (1 + (tau * omega0) ** 2)) *
            (tau * omega0 * np.sin(omega0 * t) + np.cos(omega0 * t))
        ])

        return {
            'inputs': inputs,
            'outputs': outputs,
            'dt': dt
        }

    def perturb_params(self):
        """Perturb true parameters of a dynamical system."""
        dev_fractions = self.param_uncertainty
        perturbations = np.random.uniform(low=-dev_fractions, high=dev_fractions)
        perturbed_params = (perturbations + 1.0) * self.true_params
        return perturbed_params

    @property
    def true_params(self):
        # return np.array([(10**11) * (27 * 10**(-12))])
        return np.array([2.7 * 10**(-1)])

    @property
    def param_uncertainty(self):
        return np.array([0.5])

    @property
    def system_name(self):
        return 'circuit1'

    @property
    def params_names(self):
        return ['\\tau']

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 25.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix()

