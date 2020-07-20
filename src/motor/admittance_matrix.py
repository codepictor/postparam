import sympy


class AdmittanceMatrix:

    def __init__(self, L, R, J, K):
        """Define relations between input and output data."""
        b = sympy.symbols('b', real=True)
        omega = sympy.symbols('omega', real=True)

        self._params = [b]
        self._omega = omega
        self._admittance_matrix = sympy.Matrix([
            K / ((sympy.I * omega * L + R) * (b + sympy.I * omega * J) + K**2)
        ])

    @property
    def params(self):
        """sympy.symbols which are parameters of a dynamical system."""
        return self._params

    @property
    def omega(self):
        """sympy.symbol which represents symbol of frequency."""
        return self._omega

    @property
    def data(self):
        """Raw sympy.Matrix that is true admittance matrix of a system."""
        return self._admittance_matrix

