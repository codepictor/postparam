import sympy


class AdmittanceMatrix:

    def __init__(self, L, R, J):
        """Define the relation between input and output data."""
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
        """Parameters of a dynamical system (list of sympy.symbols)."""
        return self._params

    @property
    def omega(self):
        """Frequency (sympy.symbol)."""
        return self._omega

    @property
    def data(self):
        """Admittance matrix of a system (sympy.Matrix)."""
        return self._admittance_matrix

