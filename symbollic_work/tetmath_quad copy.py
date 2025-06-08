import sympy as sp

def gradient(expr):
    return sp.Matrix([sp.diff(expr, x), sp.diff(expr, y), sp.diff(expr, z)])



a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
c1, c2, c3, c4 = sp.symbols('c1 c2 c3 c4')
d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')

x,y,z, V = sp.symbols('x y z V')

Em1, Em2 = sp.symbols('Em1 Em2')

x1, x2, x3 = sp.symbols('x1 x2 x3')
y1, y2, y3 = sp.symbols('y1 y2 y3')
z1, z2, z3 = sp.symbols('z1 z2 z3')

sp.pprint(sp.inv_quick(sp.Matrix([[1, x1, y1],[1, x2, y2],[1, x3, y3]])))

######### EDGE MODES

L1 = (a1 + b1*x + c1*y + d1*z)/(6*V)
L2 = (a2 + b2*x + c2*y + d2*z)/(6*V)

Wij1 = L1*(L2 * gradient(L1) - L1 * gradient(L2))
Wij2 = L2*(L2 * gradient(L1) - L1 * gradient(L2))

length = sp.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

Evec1 = Em1*Wij1*length
Evec2 = Em2*Wij2*length


Evec = sp.simplify(Evec1 + Evec2)

print(f'ex = {str(Evec[0]).replace("sqrt","np.sqrt")}')
print(f'ey = {str(Evec[1]).replace("sqrt","np.sqrt")}')
print(f'ez = {str(Evec[2]).replace("sqrt","np.sqrt")}')


########## FACE MODES 

# Em1, Em2 = sp.symbols('Em1 Em2')

# L1 = (a1 + b1*x + c1*y + d1*z)/(6*V)
# L2 = (a2 + b2*x + c2*y + d2*z)/(6*V)
# L3 = (a3 + b3*x + c3*y + d3*z)/(6*V)

# Wij1 = L1*L2 * gradient(L3) - L2*L3 * gradient(L1)
# Wij2 = L2*L3 * gradient(L1) - L3*L1 * gradient(L2)

# length1 = sp.sqrt((x3-x1)**2 + (y3-y1)**2 + (z3-z1)**2)
# length2 = sp.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

# Evec1 = Em1*Wij1*length1
# Evec2 = Em2*Wij2*length2

# Evec = sp.simplify(Evec1 + Evec2)
# print(f'efx = {str(Evec[0]).replace("sqrt","np.sqrt")}')
# print(f'efy = {str(Evec[1]).replace("sqrt","np.sqrt")}')
# print(f'efz = {str(Evec[2]).replace("sqrt","np.sqrt")}')


def gradient(expr):
    return sp.Matrix([sp.diff(expr, x), sp.diff(expr, y)])



######### EDGE MODES

A = sp.symbols('Area')

a1, a2, a3 = sp.symbols('A1 A2 A3')
b1, b2, b3 = sp.symbols('B1 B2 B3')
c1, c2, c3 = sp.symbols('C1 C2 C3')

L1 = (a1 + b1*x + c1*y)/(2*A)
L2 = (a2 + b2*x + c2*y)/(2*A)

Wij1 = L1*(L2 * gradient(L1) - L1 * gradient(L2))
Wij2 = L2*(L2 * gradient(L1) - L1 * gradient(L2))

length = sp.sqrt((x2-x1)**2 + (y2-y1)**2)

length = sp.symbols('Li')
Evec1 = Wij1*length
Evec2 = Wij2*length

print(f'Ee1x = {str(Evec1[0]).replace("sqrt","np.sqrt")}')
print(f'Ee1y = {str(Evec1[1]).replace("sqrt","np.sqrt")}')
print(f'Ee2x = {str(Evec2[0]).replace("sqrt","np.sqrt")}')
print(f'Ee2y = {str(Evec2[1]).replace("sqrt","np.sqrt")}')

########## FACE MODES 

Em1, Em2 = sp.symbols('Em1 Em2')


L1 = (a1 + b1*x + c1*y)/(2*A)
L2 = (a2 + b2*x + c2*y)/(2*A)
L3 = (a3 + b3*x + c3*y)/(2*A)

Wij1 = L1*L2 * gradient(L3) - L2*L3 * gradient(L1)
Wij2 = L2*L3 * gradient(L1) - L3*L1 * gradient(L2)

length1 = sp.sqrt((x3-x1)**2 + (y3-y1)**2)
length2 = sp.sqrt((x2-x1)**2 + (y2-y1)**2)

length1 = sp.symbols('Lt1')
length2 = sp.symbols('Lt2')

Evec1 = Wij1*length1
Evec2 = Wij2*length2

print(f'Ef1x = {str(Evec1[0]).replace("sqrt","np.sqrt")}')
print(f'Ef1y = {str(Evec1[1]).replace("sqrt","np.sqrt")}')
print(f'Ef2x = {str(Evec2[0]).replace("sqrt","np.sqrt")}')
print(f'Ef2y = {str(Evec2[1]).replace("sqrt","np.sqrt")}')