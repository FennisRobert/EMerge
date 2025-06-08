import sympy as sp


############## 2D

def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y)])

x, y = sp.symbols('x y')

x1, x2, x3 = sp.symbols('x1 x2 x3')
y1, y2, y3 = sp.symbols('y1 y2 y3')
e1, e2, e3, e4, e5, e6 = sp.symbols('e1 e2 e3 e4 e5 e6')

M = sp.Matrix([[1, x1, y1],[1, x2, y2],[1, x3, y3]])

#a1, b1, c1 = sp.Inverse(M) @ sp.Matrix([[1],[0],[0]])
#a2, b2, c2 = sp.Inverse(M) @ sp.Matrix([[0],[1],[0]])
#a3, b3, c3 = sp.Inverse(M) @ sp.Matrix([[0],[0],[1]])

a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')
c1, c2, c3 = sp.symbols('c1 c2 c3')

area = sp.symbols('A')

L1 = (a1 + b1*x + c1*y)/(2*area)
L2 = (a2 + b2*x + c2*y)/(2*area)
L3 = (a3 + b3*x + c3*y)/(2*area)

f1 = e1*L1*(2*L1 - 1)
f2 = e2*L2*(2*L2 - 1)
f3 = e3*L3*(2*L3 - 1)
f4 = e4*4*L1*L2
f5 = e5*4*L2*L3
f6 = e6*4*L1*L3

Ne1 = sp.simplify(L1*L2*grad(L3) - L3*L2*grad(L1))
Ne2 = sp.simplify(L2*L3*grad(L1) - L1*L3*grad(L2))

for i, ff in enumerate([f1, f2, f3, f4, f5, f6]):
    print(f'V{i} = {sp.simplify(ff)}')



################## 3D


def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y), sp.diff(Lf, z)])

x, y, z = sp.symbols('x y z')

x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3, y4 = sp.symbols('y1 y2 y3 y4')
z1, z2, z3, z4 = sp.symbols('z1 z2 z3 z4')

e1, e2, e3, e4, e5, e6, e7, e8, e9, e10 = sp.symbols('e1 e2 e3 e4 e5 e6 e7 e8 e9 e10')

M = sp.Matrix([[1, x1, y1, z1],[1, x2, y2, z2],[1, x3, y3, z3],[1, x4, y4, z4]])

#a1, b1, c1 = sp.Inverse(M) @ sp.Matrix([[1],[0],[0]])
#a2, b2, c2 = sp.Inverse(M) @ sp.Matrix([[0],[1],[0]])
#a3, b3, c3 = sp.Inverse(M) @ sp.Matrix([[0],[0],[1]])

a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
c1, c2, c3, c4 = sp.symbols('c1 c2 c3 c4')
d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')

V = sp.symbols('V')

L1 = (a1 + b1*x + c1*y + d1*z)/(6*V)
L2 = (a2 + b2*x + c2*y + d2*z)/(6*V)
#L3 = (a3 + b3*x + c3*y + d3*z)/(6*V)
#L4 = (a4 + b4*x + c4*y + d4*z)/(6*V)



fv = e1*L1*(2*L1 - 1)

fe  = e2*4*L1*L2

print(f'dvdx = {sp.simplify(sp.diff(fv,x))}')
print(f'dvdy = {sp.simplify(sp.diff(fv,y))}')
print(f'dvdz = {sp.simplify(sp.diff(fv,z))}')

print(f'dedx = {sp.simplify(sp.diff(fe,x))}')
print(f'dedy = {sp.simplify(sp.diff(fe,y))}')
print(f'dedz = {sp.simplify(sp.diff(fe,z))}')
