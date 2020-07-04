import numpy as np

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        """Simulate data in time domain."""
        epsilon0 = 2
        omega0 = 2 * np.pi * 50  # 50 Hz
        tau = self.true_params[0]

        min_t = 0.0
        max_t = 10.0
        dt = 0.001
        points_n = int((max_t - min_t) / dt)

        t = np.linspace(0, 10, num=points_n)
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
        dev_fractions = np.array([1.0])
        perturbations = np.random.uniform(low=-dev_fractions, high=dev_fractions)
        perturbed_params = (perturbations + 1.0) * self.true_params
        return perturbed_params, dev_fractions

    @property
    def true_params(self):
        return np.array([(10**4) * (27 * 10**(-12))])

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

