import sympy as sp


############## 2D
x, y, z = sp.symbols('x y z')

def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y), sp.diff(Lf, z)])

def curl(A):
    return sp.Matrix([sp.diff(A[2], y) - sp.diff(A[1], z),
                      sp.diff(A[0], z) - sp.diff(A[2], x),
                      sp.diff(A[1], x) - sp.diff(A[0], y)])



e1, e2 = sp.symbols('Em1 Em2')

a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')
c1, c2, c3 = sp.symbols('c1 c2 c3')
d1, d2, d3 = sp.symbols('d1 d2 d3')

volume = sp.symbols('V')
L = sp.symbols('L')
L1 = (a1 + b1*x + c1*y + d1*z)/(6*volume)
L2 = (a2 + b2*x + c2*y + d2*z)/(6*volume)
L3 = (a3 + b3*x + c3*y + d3*z)/(6*volume)

Ne1 = sp.simplify(L1*(L2*grad(L1) - L1*grad(L2)))*L
Ne2 = sp.simplify(L2*(L2*grad(L1) - L1*grad(L2)))*L

print('ex = ', sp.simplify(e1*Ne1[0] + e2*Ne2[0]))
print('ey = ', sp.simplify(e1*Ne1[1] + e2*Ne2[1]))
print('ez = ', sp.simplify(e1*Ne1[2] + e2*Ne2[2]))

El1, El2 = sp.symbols('L1 L2')
Nf1 = sp.simplify(L1*L2*grad(L3) - L2*L3*grad(L1))*El1
Nf2 = sp.simplify(L2*L3*grad(L1) - L3*L1*grad(L2))*El2

print('ex = ', sp.simplify(e1*Nf1[0] + e2*Nf2[0]))
print('ey = ', sp.simplify(e1*Nf1[1] + e2*Nf2[1]))
print('ez = ', sp.simplify(e1*Nf1[2] + e2*Nf2[2]))

print('')

################## Curls

Ne1 = curl(sp.simplify(L1*(L2*grad(L1) - L1*grad(L2)))*L)
Ne2 = curl(sp.simplify(L2*(L2*grad(L1) - L1*grad(L2)))*L)

print('ex = ', sp.simplify(e1*Ne1[0] + e2*Ne2[0]))
print('ey = ', sp.simplify(e1*Ne1[1] + e2*Ne2[1]))
print('ez = ', sp.simplify(e1*Ne1[2] + e2*Ne2[2]))

El1, El2 = sp.symbols('L1 L2')
Nf1 = curl(sp.simplify(L1*L2*grad(L3) - L2*L3*grad(L1))*El1)
Nf2 = curl(sp.simplify(L2*L3*grad(L1) - L3*L1*grad(L2))*El2)

print('ex = ', sp.simplify(e1*Nf1[0] + e2*Nf2[0]))
print('ey = ', sp.simplify(e1*Nf1[1] + e2*Nf2[1]))
print('ez = ', sp.simplify(e1*Nf1[2] + e2*Nf2[2]))