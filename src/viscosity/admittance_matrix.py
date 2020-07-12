import sympy


class AdmittanceMatrix:

    def __init__(self):
        """Define relations between input and output data."""
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

