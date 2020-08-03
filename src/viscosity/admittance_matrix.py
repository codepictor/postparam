import sympy


class AdmittanceMatrix:

    def __init__(self):
        """Define the relation between input and output data."""
        m = sympy.symbols('m', real=True)
        B = sympy.symbols('B', real=True)
        omega = sympy.symbols('omega', real=True)

        self._params = [m, B]
        self._omega = omega
        self._admittance_matrix = sympy.Matrix([
            [1 / (B + sympy.I * omega * m), 0],
            [0, 1 / (B + sympy.I * omega * m)]
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
        """Admittance matrix of a system (sympy.Matrix)."""
        return self._admittance_matrix

