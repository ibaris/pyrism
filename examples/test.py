from sympy import symbols, I
from sympy.physics.quantum import Ket, Bra

s12, s1, s2 = symbols("s12, -s1, -s2")
e, e1, e2 = symbols("e, e1, e2")

s = Ket(1, 2, 3)
E = Bra(e, e1, e2)
