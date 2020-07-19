import numpy as np

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):


        return {
            'inputs': ,
            'outputs': ,
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
        return np.array([])

    @property
    def param_uncertainty(self):
        return np.array([])

    @property
    def system_name(self):
        return 'motor'

    @property
    def params_names(self):
        return []

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 8.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix()

