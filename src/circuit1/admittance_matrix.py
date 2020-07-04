import sympy


class AdmittanceMatrix:

    def __init__(self):
        """Define relations between input and output data."""
        tau = sympy.symbols('tau', real=True)
        omega = sympy.symbols('omega', real=True)

        self._params = [tau]
        self._omega = omega
        self._admittance_matrix = sympy.Matrix([
            1 / (1 + sympy.I * omega * tau)
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

