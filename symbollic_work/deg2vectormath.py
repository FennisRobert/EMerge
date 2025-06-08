import sympy as sp


def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y)])

def curl(A, B):
    
x, y = sp.symbols('x y')

x1, x2, x3 = sp.symbols('x1 x2 x3')
y1, y2, y3 = sp.symbols('y1 y2 y3')
f1, f2, f3 = sp.symbols('f1 f2 f3')

M = sp.Matrix([[1, x1, y1],[1, x2, y2],[1, x3, y3]])

a1, b1, c1 = sp.Inverse(M) @ sp.Matrix([[1],[0],[0]])
a2, b2, c2 = sp.Inverse(M) @ sp.Matrix([[0],[1],[0]])
a3, b3, c3 = sp.Inverse(M) @ sp.Matrix([[0],[0],[1]])

area = sp.symbols('A')
L1 = (a1 + b1*x + c1*y)/(2*area)
L2 = (a2 + b2*x + c2*y)/(2*area)
L3 = (a3 + b3*x + c3*y)/(2*area)


Ne1 = sp.simplify(L1*L2*grad(L3) - L3*L2*grad(L1))
Ne2 = sp.simplify(L2*L3*grad(L1) - L1*L3*grad(L2))

print(f'Ne1x = {sp.simplify(Ne1[0])}')
print(f'Ne1y = {sp.simplify(Ne1[1])}')
print(f'Ne2x = {sp.simplify(Ne2[0])}')
print(f'Ne2y = {sp.simplify(Ne2[1])}')

