import sympy


class AdmittanceMatrix:

    def __init__(self, L, R, J):
        """Define relations between input and output data."""
        b = sympy.symbols('b', real=True)
        K = sympy.symbols('K', real=True)
        omega = sympy.symbols('omega', real=True)

        self._params = [b, K]
        self._omega = omega
        self._admittance_matrix = (
            1 / ((R + sympy.I * omega * L) * (b + sympy.I * omega * J) + K**2)
            * sympy.Matrix([
                [K, -R - sympy.I * omega * L],
                [b + sympy.I * omega * J, K]
            ])
        )

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

