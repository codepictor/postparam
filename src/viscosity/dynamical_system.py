import numpy as np

from . import admittance_matrix


class DynamicalSystem:

    def generate_time_data(self):
        F0 = 3
        omega0 = 2 * np.pi * 0.5  # 0.5 Hz
        m = self.true_params[0]
        B = self.true_params[1]

        min_t = 0.0
        max_t = 50.0
        dt = 0.2
        t = np.arange(min_t, max_t, step=dt)

        Fx = F0 * (1 + np.sin(omega0 * t))
        Fy = F0 * np.cos(2 * omega0 * t)

        vx0 = 0
        vx = (
            (
                B**2 * F0
                + B**2 * F0 * np.sin(omega0 * t)
                + F0 * m**2 * omega0**2
                - B * F0 * m * omega0 * np.cos(omega0 * t)
            ) /
            (
                B**3 + B * m**2 * omega0**2
            ) +
            (
                vx0 + F0 * ((m * omega0) / (B**2 + m**2 * omega0**2) - 1 / B)
            ) *
            (
                np.exp(-B / m * t)
            )
        )

        vy0 = 0
        vy = (
            (
                2 * F0 * m * omega0 * np.sin(2 * omega0 * t)
                + B * F0 * np.cos(2 * omega0 * t)
            ) /
            (
                B**2 + 4 * m**2 * omega0**2
            ) +
            (
                vy0 - B * F0 / (B**2 + 4 * m**2 * omega0**2)
            ) *
            (
                np.exp(-B / m * t)
            )
        )

        return {
            'inputs': np.array([Fx, Fy]),
            'outputs': np.array([vx, vy]),
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
        return np.array([10.0, 20.0])  # m, B

    @property
    def param_uncertainty(self):
        return np.array([0.5, 0.5])

    @property
    def system_name(self):
        return 'viscosity'

    @property
    def params_names(self):
        return ['m', 'B']

    @property
    def inputs_names(self):
        return ['F_x', 'F_y']

    @property
    def outputs_names(self):
        return ['v_x', 'v_y']

    @property
    def min_freq(self):
        return 0.0

    @property
    def max_freq(self):
        return 8.0

    @property
    def admittance_matrix(self):
        return admittance_matrix.AdmittanceMatrix()

