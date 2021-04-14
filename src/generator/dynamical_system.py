import numpy as np

from . import admittance_matrix
from . import dynamic_equations_to_simulate


class DynamicalSystem:

    def generate_time_data(self):
        ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
            noise={
                'rnd_amp': 0.002
            },
            gen_param={
                'd_2': self.true_params[0],
                'e_2': self.true_params[1],
                'm_2': self.true_params[2],
                'x_d2': self.true_params[3],
                'ic_d2': 1.0
            },
            osc_param={
                'osc_amp': 0.000,
                'osc_freq': 0.0
            },
            integr_param={
                'df_length': 100.0,
                'dt_step': 0.05
            }
        )

        ode_solver_object.simulate_time_data()
        inputs = np.array([
            ode_solver_object.Vc1_abs - ode_solver_object.Vc1_abs.mean(),
            ode_solver_object.Vc1_angle - ode_solver_object.Vc1_angle.mean()
        ])
        outputs = np.array([
            ode_solver_object.Ig_abs - ode_solver_object.Ig_abs.mean(),
            ode_solver_object.Ig_angle - ode_solver_object.Ig_angle.mean()
        ])

        return {
            'inputs': inputs,
            'outputs': outputs,
            'dt': ode_solver_object.dt
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
        return np.array([
            0.25, 1.00, 1.00, 0.01
        ])

    @property
    def param_uncertainty(self):
        return np.array([
            0.5, 0.5, 0.5, 0.5
        ])

    @property
    def system_name(self):
        return 'generator'

    @property
    def params_names(self):
        return ['D', 'E', 'M', 'X']

    @property
    def inputs_names(self):
        return ['abs(V)', 'angle(V)']

    @property
    def outputs_names(self):
        return ['abs(I)', 'angle(I)']

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 6.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix()

