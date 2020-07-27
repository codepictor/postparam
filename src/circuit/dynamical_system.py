import numpy as np

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        epsilon0 = 3
        omega0 = 2 * np.pi * 4  # 4 Hz
        tau = self.true_params[0]

        min_t = 0.0
        max_t = 30.0
        dt = 0.05

        t = np.arange(min_t, max_t, step=dt)
        E = epsilon0 * np.cos(omega0 * t)
        V = (
            (
                epsilon0 / (1 + (tau * omega0)**2)
            ) *
            (
                tau * omega0 * np.sin(omega0 * t)
                + np.cos(omega0 * t)
                - np.exp(-t / tau)
            )
        )

        return {
            'inputs': np.array([E]),
            'outputs': np.array([V]),
            'dt': dt
        }

    def perturb_true_params(self):
        perturbations = np.random.uniform(
            low=-self.param_uncertainty,
            high=self.param_uncertainty
        )
        perturbed_params = (perturbations + 1.0) * self.true_params
        return perturbed_params

    @property
    def true_params(self):
        # return np.array([(10 * 10**3) * (2.7 * 10**(-6))])
        return np.array([2.7 * 10**(-2)])  # tau

    @property
    def param_uncertainty(self):
        return np.array([0.5])

    @property
    def system_name(self):
        return 'circuit'

    @property
    def params_names(self):
        return ['\\tau']

    @property
    def inputs_names(self):
        return ['\\varepsilon']

    @property
    def outputs_names(self):
        return ['V']

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 8.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix()

