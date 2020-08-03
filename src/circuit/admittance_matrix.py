import sympy


class AdmittanceMatrix:

    def __init__(self):
        """Define the relation between input and output data."""
        tau = sympy.symbols('tau', real=True)
        omega = sympy.symbols('omega', real=True)

        self._params = [tau]
        self._omega = omega
        self._admittance_matrix = sympy.Matrix([
            1 / (1 + sympy.I * omega * tau)
        ])

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
        """Admittance matrix of a dynamical system (sympy.Matrix)."""
        return self._admittance_matrix

