from sympy import Abs, Determinant, Matrix, diff, integrate, pprint, simplify, symbols, exp, sqrt

GENERATE = True

x1, x2, x3, x4, x5, x = symbols("x1 x2 x3 x4 x5 x")
a, b, q, y, f = symbols("a b q y f")


#############################################
############### First Order #################
#############################################

if False:
    order = 1
    Npe = order + 1
    M = Matrix([[x1, 1], [x2, 1]])
    iM = simplify(M.inv())
    xvec = Matrix([[x], [1]])
    fs = iM.T @ xvec
    print(fs)
    print(
        f"        Kmat[i,i] += {simplify(integrate(a * diff(fs[0], x) * diff(fs[0], x) + b * fs[0] * fs[0],(x, x1, x2)))}"
    )
    print(
        f"        Kmat[i,j] += {simplify(integrate(a * diff(fs[0], x) * diff(fs[1], x) + b * fs[0] * fs[1],(x, x1, x2)))}"
    )
    print(
        f"        Kmat[j,i] += {simplify(integrate(a * diff(fs[1], x) * diff(fs[0], x) + b * fs[1] * fs[0],(x, x1, x2)))}"
    )
    print(
        f"        Kmat[j,j] += {simplify(integrate(a * diff(fs[1], x) * diff(fs[1], x) + b * fs[1] * fs[1],(x, x1, x2)))}"
    )

    print(f"        bmat[i] += {integrate(fs[0]*f,(x,x1,x2))}")
    print(f"        bmat[j] += {integrate(fs[1]*f,(x,x1,x2))}")

if False:
    x1, x2, x3, y1, y2, y3 = symbols("x1 x2 x3 y1 y2 y3")
    m11, m12, m13, m21, m22, m23, m31, m32, m33 = symbols(
        "m11 m12 m13 m21 m22 m23 m31 m32 m33"
    )
    x, y = symbols("x y")
    t1, t2 = symbols("t1 t2")
    M1 = Matrix([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
    M = M1.inv().T

    c = Matrix([[x], [y], [1]])

    fs = M @ c
    f1 = fs[0]
    f2 = fs[1]
    f3 = fs[2]
    pprint(f1)
    pprint(f2)
    pprint(f3)

    IJ = [(i, j) for i in range(3) for j in range(3)]

    d1x = x2 - x1
    d2x = x3 - x1
    d1y = y2 - y1
    d2y = y3 - y1

    xn = d1x * t1 + d2x * t2 + x1
    yn = d1y * t1 + d2y * t2 + y1
    Area = 0.5 * Abs(Determinant(M))
    A = symbols("A")
    syms = ["i", "j", "k"]
    for i, j in IJ:
        Fi = fs[i].subs(x, xn).subs(y, yn)
        Fj = fs[j].subs(x, xn).subs(y, yn)
        F = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        print(f"T[{syms[i]},{syms[j]}] += 0.5*A*{simplify(F)}")
    fs_b = fs
    ax, ay, beta = symbols("ax ay beta")
    M = Matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    fs = M @ c
    for i, j in IJ:
        Fi = diff(fs[i], x).subs(x, xn).subs(y, yn)
        Fj = diff(fs[j], x).subs(x, xn).subs(y, yn)
        F1 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        Fi = diff(fs[i], y).subs(x, xn).subs(y, yn)
        Fj = diff(fs[j], y).subs(x, xn).subs(y, yn)
        F2 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        Fi = fs_b[i].subs(x, xn).subs(y, yn)
        Fj = fs_b[j].subs(x, xn).subs(y, yn)
        F3 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        print(
            f"D[{syms[i]},{syms[j]}] += 2*A*{simplify(ax*F1)} + 2*A*{simplify(ay*F2)} + 2*A*{simplify(beta*F3)}"
        )



y = symbols('gamma')


t, d, d1, l = symbols('t d d1 l')

d2 = d1 + l

Ml = Matrix([[d1, 1],[d2, 1]])
Ml2 = Ml.inv().T
c = Matrix([[d],[1]])

fs = simplify(Ml2 @ c)

for i in range(2):
    pprint(integrate(fs[i].subs(d,d1+t*(d2-d1)), (t,0,1)))

print('Linear line integral solution')
symbs = symbols('i j')
for i,j in [(i,j) for i in range(2) for j in range(2)]:
    print(f'D[{symbs[i]},{symbs[j]}] += -1j * beta * length * {integrate((fs[i]*fs[j]).subs(d,d1+t*(d2-d1)), (t,0,1))}')

print('linear abc solution')

y1, y2 = symbols('gamma1 gamma2')
for i,j in [(i,j) for i in range(2) for j in range(2)]:
    integrand = (y1*l*fs[i]*fs[j]).subs(d,d1+t*l) - y2/l*diff(fs[i].subs(d,d1+t*l),t)*diff(fs[j].subs(d,d1+t*l),t)
    res = integrate(integrand, (t,0,1))
    res = simplify(res.subs(d1,0).subs(d2,d1+l))
    print(f'D[{symbs[i]},{symbs[j]}] += {res}')
    

print()
print()

print('QUADRATIC')
d2 = d1 + l
dc = d1 + l/2

M = Matrix([[d1**2, d1, 1],[dc**2, dc, 1],[d2**2, d2, 1]])
M2 = M.inv().T

c = Matrix([[d**2],[d],[1]])

fs = simplify(M2 @ c)


for i in range(3):
    pprint(integrate(fs[i].subs(d,d1+t*(d2-d1)), (t,0,1)))

symbs = 'ikj'
for i,j in [(i,j) for i in range(3) for j in range(3)]:

    print(f'D[{symbs[i]},{symbs[j]}] += -1j * beta * length * {integrate((fs[i]*fs[j]).subs(d,d1+t*(d2-d1)), (t,0,1))}')

print('quadratic abc solution first')

l, y1, y2 = symbols('l gamma1 gamma2')
for i,j in [(i,j) for i in range(3) for j in range(3)]:
    res = integrate((y*l*fs[i]*fs[j]).subs(d,d1+t*(d2-d1)), (t,0,1))
    res = simplify(res.subs(d1,0).subs(d2,l))
    print(f'D[{symbs[i]},{symbs[j]}] += {res}')

print('quadratic abc solution second order')

l, y1, y2 = symbols('l gamma1 gamma2')
for i,j in [(i,j) for i in range(3) for j in range(3)]:
    F1 = (fs[i]*fs[j])#.subs(d,d1+t*l)
    #F2 = diff(fs[i].subs(d,d1+t*l),t)
    #F3 = diff(fs[j].subs(d,d1+t*l),t)
    F2 = diff(fs[i],d)
    F3 = diff(fs[j],d)
    integrand = y1*F1 - y2*F2*F3
    res = l*integrate(integrand.subs(d,d1+t*l), (t,0,1))
    res = simplify(res)
    #res = simplify(res.subs(d1,0))
    print(f'D[{symbs[i]},{symbs[j]}] += {res}')


print("Derivative with respect to normals and edge")

k0, x, x0, y, y0 = symbols('k0 x x0 y y0')
psi = exp(-1j*k0*sqrt((x-x0)**2 + (y-y0)**2))/sqrt((x-x0)**2 + (y-y0)**2)
psix = diff(psi,x)
psiy = diff(psi,y)
R = symbols('R')
print(f'Psi x = {simplify((psix/psi).subs(sqrt((x - x0)**2 + (y - y0)**2),R))}')
print(f'Psi y = {simplify((psiy/psi).subs(sqrt((x - x0)**2 + (y - y0)**2),R))}')

psix = diff(psix,x)
psiy = diff(psiy,y)

pprint(simplify((psix/psi).subs(sqrt((x - x0)**2 + (y - y0)**2),R)))
pprint(simplify((psiy/psi).subs(sqrt((x - x0)**2 + (y - y0)**2),R)))

quit()
if False:
    x1, x2, x3, y1, y2, y3 = symbols("x1 x2 x3 y1 y2 y3")

    x4 = (x1+x2)/2
    x5 = (x2+x3)/2
    x6 = (x1+x3)/2
    y4 = (y1+y2)/2
    y5 = (y2+y3)/2
    y6 = (y1+y3)/2

    Msymbs = symbols(' '.join([f'm{i+1}{j+1}' for i in range(6) for j in range(6)]))
    Msymbs = [[Msymbs[i+j*6] for i in range(6)] for j in range(6)]

    x, y = symbols("x y")
    t1, t2 = symbols("t1 t2")
    M1 = Matrix([[a**2, b**2, a*b, a, b, 1] for a,b in zip([x1, x2, x3, x4, x5, x6],[y1, y2, y3, y4, y5, y6])])


    M = M1.inv().T

    c = Matrix([[x**2], [y**2],[x*y],[x],[y],[1]])

    fs = simplify(M @ c)
    f1 = fs[0]
    f2 = fs[1]
    f3 = fs[2]
    f4 = fs[3]
    f5 = fs[4]
    f6 = fs[5]

    print(fs)
    # quit()

    IJ = [(i, j) for i in range(6) for j in range(6)]

    d1x = x2 - x1
    d2x = x3 - x1
    d1y = y2 - y1
    d2y = y3 - y1

    xn = d1x * t1 + d2x * t2 + x1
    yn = d1y * t1 + d2y * t2 + y1
    
    MA = Matrix([[x]])
    Area = 0.5 * Abs(Determinant(M))
    #A = symbols("A")
    syms = ["i1","j1","k1","i2","j2","k2"]
    #for i, j in IJ:
    #    Fi = fs[i].subs(x, xn).subs(y, yn)
    #    Fj = fs[j].subs(x, xn).subs(y, yn)
    #    F = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
    #    print(f"T[{syms[i]},{syms[j]}] += 0.5*A*{simplify(F)}")
    fs_b = fs
    ax, ay, beta = symbols("ax ay beta")
    #M = Matrix(Msymbs)
    #pprint(M)
    
    
    #fs = M @ c
    pprint(fs)
    for i, j in IJ:
        Fi = diff(fs[i], x).subs(x, xn).subs(y, yn)
        Fj = diff(fs[j], x).subs(x, xn).subs(y, yn)
        F1 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        Fi = diff(fs[i], y).subs(x, xn).subs(y, yn)
        Fj = diff(fs[j], y).subs(x, xn).subs(y, yn)
        F2 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        Fi = fs_b[i].subs(x, xn).subs(y, yn)
        Fj = fs_b[j].subs(x, xn).subs(y, yn)
        F3 = integrate(integrate(Fi * Fj, (t2, 0, 1 - t1)), (t1, 0, 1))
        print(
            f"D[{syms[i]},{syms[j]}] += 2*A*{simplify(ax*F1+ay*F2+beta*F3)}"
        )
