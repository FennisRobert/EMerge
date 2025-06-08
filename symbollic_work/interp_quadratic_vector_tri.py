import sympy as sp


############## 2D
x, y, z = sp.symbols('x y z')
jB = sp.symbols('jB')

def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y), 0])

def curl(Lf):
    return sp.Matrix([sp.diff(Lf[2], y) + jB*Lf[1],
                      sp.diff(Lf[2], x) - jB*Lf[0],
                      sp.diff(Lf[1], x) - sp.diff(Lf[0], y)])

e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14 = sp.symbols('e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 e12 e13 e14')

a1, a2, a3 = sp.symbols('a1 a2 a3')
b1, b2, b3 = sp.symbols('b1 b2 b3')
c1, c2, c3 = sp.symbols('c1 c2 c3')

area = sp.symbols('A')
L = sp.symbols('L')
L1 = (a1 + b1*x + c1*y)/(2*area)
L2 = (a2 + b2*x + c2*y)/(2*area)
L3 = (a3 + b3*x + c3*y)/(2*area)

def Ne1(l1, l2):
    return sp.simplify((l1*(l2*grad(l1) - l1*grad(l2))))
def Ne2(l1, l2):
    return sp.simplify((l2*(l2*grad(l1) - l1*grad(l2))))
Nf1 = sp.simplify(L1*L2*grad(L3) - L2*L3*grad(L1))
Nf2 = sp.simplify(L2*L3*grad(L1) - L3*L1*grad(L2))
def Nlv(l1):
    return sp.Matrix([0, 0, sp.simplify((2*L1*L1 - L1))])
def Nle(l1, l2):
    return sp.Matrix([0, 0, sp.simplify((2*L1*L2))])

E = e1*Ne1(L1, L2) + e2*Ne1(L2,L3) + e3*Ne1(L1,L3) + e4*Nf1 + e5*Ne2(L1,L2) + e6*Ne2(L2,L3) + e7*Ne2(L1,L3) + e8*Nf2
E = E + e9*Nlv(L1) + e10*Nlv(L2) + e11*Nlv(L3) + e12*Nle(L1,L2) + e13*Nle(L2,L3) + e14*Nle(L1,L3)
E = sp.simplify(E)

print('ex = ', E[0])
print('ey = ', E[1])
print('ez = ', E[2])

H = sp.simplify(curl(E))
print('')

print('hx = ', H[0])
print('hy = ', H[1])
print('hz = ', H[2])

