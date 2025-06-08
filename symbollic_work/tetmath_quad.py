import sympy as sp

x1, x2, x3, x4  = sp.symbols('x1 x2 x3 x4')
y1, y2, y3, y4  = sp.symbols('y1 y2 y3 y4')
z1, z2, z3, z4  = sp.symbols('z1 z2 z3 z4')
f1, f2, f3, f4  = sp.symbols('f1 f2 f3 f4')

V = sp.symbols('V')

aes = sp.det(sp.Matrix([[f1, f2, f3, f4],[x1, x2, x3, x4],[y1, y2, y3, y4],[z1, z2, z3, z4]]))
bes = sp.det(sp.Matrix([[1, 1, 1, 1],[f1, f2, f3, f4],[y1, y2, y3, y4],[z1, z2, z3, z4]]))
ces = sp.det(sp.Matrix([[1, 1, 1, 1],[x1, x2, x3, x4],[f1, f2, f3, f4],[z1, z2, z3, z4]]))
des = sp.det(sp.Matrix([[1, 1, 1, 1],[x1, x2, x3, x4],[y1, y2, y3, y4],[f1, f2, f3, f4]]))

fs = [f1, f2, f3, f4]

#V = sp.simplify(sp.det(sp.Matrix([[1,1,1,1],[x1,x2,x3,x4],[y1,y2,y3,y4],[z1,z2,z3,z4]]))/6)

# for coeff, coeffset in zip(['a','b','c','d'],[aes, bes, ces, des]):
#     for i, f in enumerate(fs):
#         cf = sp.simplify(sp.collect(sp.expand(coeffset), f).coeff(f,1))
#         print(f'{coeff}{i+1} = {cf}')
    
avs = sp.symbols('a1 a2 a3 a4')
bvs = sp.symbols('b1 b2 b3 b4')
cvs = sp.symbols('c1 c2 c3 c4')
dvs = sp.symbols('d1 d2 d3 d4')

pairs = [(i-1,j-1) for i,j in zip([1,1,1,2,4,3],[2,3,4,3,2,4])]

x,y,z = sp.symbols('x y z')

M = sp.Matrix([[1, x1, y1, z1],[1, x2, y2, z2],[1, x3, y3, z3],[1, x4, y4, z4]])

fvec = sp.Matrix([[f1],[f2],[f3],[f4]])

xs = [x1, x2, x3, x4]
ys = [y1, y2, y3, y4]
zs = [z1, z2, z3, z4]

def gradient(expr):
    return sp.Matrix([sp.diff(expr, x), sp.diff(expr, y), sp.diff(expr, z)])

e1, e2, e3, e4, e5, e6 = sp.symbols('e1 e2 e3 e4 e5 e6')
e12, e22, e32, e42, e52, e62 = sp.symbols('e12 e22 e32 e42 e52 e62')

Evec1 = 0
Evec2 = 0

iii = 0

for (i,j), ei, ei2 in zip(pairs, [e1, e2, e3, e4, e5, e6], [e12, e22, e32, e42, e52, e62]):
    L1 = (avs[i] + bvs[i]*x + cvs[i]*y + dvs[i]*z)/(6*V)
    L2 = (avs[j] + bvs[j]*x + cvs[j]*y + dvs[j]*z)/(6*V)
    Wij1 = L1*(L2 * gradient(L1) - L1 * gradient(L2))
    Wij2 = L2*(L2 * gradient(L1) - L1 * gradient(L2))
    length = sp.sqrt((xs[j]-xs[i])**2 + (ys[j]-ys[i])**2 + (zs[j]-zs[i])**2)
    #print(sp.simplify(sp.expand(Wij)))
    if iii == 0:
        Evec1 = ei*Wij1*length
        Evec2 = ei2*Wij2*length
    else:
        Evec1 = Evec1 + ei*Wij1*length
        Evec2 = Evec2 + ei2*Wij2*length
    iii += 1

Evec = sp.simplify(Evec1 + Evec2)
print(f'ex = {str(Evec[0]).replace("sqrt","np.sqrt")}')
print(f'ey = {str(Evec[1]).replace("sqrt","np.sqrt")}')
print(f'ez = {str(Evec[2]).replace("sqrt","np.sqrt")}')
##########

f1, f2, f3, f4 = sp.symbols('f1 f2 f3 f4')
f12, f22, f32, f42 = sp.symbols('f12 f22 f32 f42')

facepairs = ((0,1,2),
             (0,2,3),
             (0,3,1),
             (1,2,3))
iii= 0

for (i,j,k), ei, ei2 in zip(facepairs, [f1, f2, f3, f4], [f12, f22, f32, f42]):

    L1 = (avs[i] + bvs[i]*x + cvs[i]*y + dvs[i]*z)/(6*V)
    L2 = (avs[j] + bvs[j]*x + cvs[j]*y + dvs[j]*z)/(6*V)
    L3 = (avs[k] + bvs[k]*x + cvs[k]*y + dvs[k]*z)/(6*V)

    Wij1 = L1*L2 * gradient(L3) - L2*L3 * gradient(L1)
    Wij2 = L2*L3 * gradient(L1) - L3*L1 * gradient(L2)

    length1 = sp.sqrt((xs[k]-xs[i])**2 + (ys[k]-ys[i])**2 + (zs[k]-zs[i])**2)
    length2 = sp.sqrt((xs[j]-xs[i])**2 + (ys[j]-ys[i])**2 + (zs[j]-zs[i])**2)
    
    #area = sp.Abs(e1.cross(e2).norm())/2
    #print(sp.simplify(sp.expand(Wij)))
    if iii == 0:
        Evec1 = ei*Wij1*length1
        Evec2 = ei2*Wij2*length2
    else:
        Evec1 = Evec1 + ei*Wij1*length1
        Evec2 = Evec2 + ei2*Wij2*length2
    iii += 1

Evec = sp.simplify(Evec1 + Evec2)
print(f'efx = {str(Evec[0]).replace("sqrt","np.sqrt")}')
print(f'efy = {str(Evec[1]).replace("sqrt","np.sqrt")}')
print(f'efz = {str(Evec[2]).replace("sqrt","np.sqrt")}')