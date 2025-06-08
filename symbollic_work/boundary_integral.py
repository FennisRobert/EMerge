import sympy as sp

x, y, z = sp.symbols('x y z')
lam1, lam2, lam3 = sp.symbols('L1 L2 L3')

def grad(expr):
    return sp.Matrix([sp.diff(expr, x), sp.diff(expr, y)])

def crossz(v):
    return sp.Matrix([-v[1], v[0]])


x1, x2, x3 = sp.symbols('x1 x2 x3')
y1, y2, y3 = sp.symbols('y1 y2 y3')

f1, f2, f3 = sp.symbols('f1 f2 f3')
U1x, U2x, U3x = sp.symbols('U1x U2x U3x')
U1y, U2y, U3y = sp.symbols('U1y U2y U3y')

U1 = sp.Matrix([U1x, U1y])
U2 = sp.Matrix([U2x, U2y])
U3 = sp.Matrix([U3x, U3y])

l1, l2, l3 = sp.symbols('l1 l2 l3')


area = sp.symbols('Area')


a1, a2, a3, b1, b2, b3, c1, c2, c3 = sp.symbols('a1 a2 a3 b1 b2 b3 c1 c2 c3')

L1 = (a1 + b1*x + c1*y)/(2*area)
L2 = (a2 + b2*x + c2*y)/(2*area)
L3 = (a3 + b3*x + c3*y)/(2*area)


N1 = (L1*grad(L2) - L2*grad(L1))
N2 = (L2*grad(L3) - L3*grad(L2))
N3 = (L3*grad(L1) - L1*grad(L3))

N1 = sp.simplify(N1*l1)
N2 = sp.simplify(N2*l2)
N3 = sp.simplify(N3*l3)

print(N1)

x1f = x1 + (x2-x1)*lam1 + (x3-x1)*lam2
y1f = y1 + (y2-y1)*lam1 + (y3-y1)*lam2

N1 = N1.subs({x: x1f, y: y1f})
N2 = N2.subs({x: x1f, y: y1f})
N3 = N3.subs({x: x1f, y: y1f})

Ns = [N1, N2, N3]
Us = [U1, U2, U3]
for i in range(3):
    I1 = sp.integrate(sp.integrate(Ns[i].dot(Us[i]), (lam1, 0, 1-lam2)), (lam2, 0, 1))
    print(f'        B{i+1} = {str(sp.simplify(-I1*2*area))}')
    print('')