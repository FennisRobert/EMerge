import sympy as sp

def product(items: list):
    result = 1
    for item in items:
        result *= item
    return result

def P(Lf, Index, n):
    elems = [(n*Lf-p)/(Index-p) for p in range(Index)]
    return product(elems)

def Phat(Lf, Index, n):
    if Index==0:
        return 1/(n*Lf)
    return Index/(n*Lf) * P(Lf, Index, n)

def grad(Lf):
    return sp.Matrix([sp.diff(Lf, x), sp.diff(Lf, y), sp.diff(Lf, z)])

def curl(Nf):
    return sp.Matrix([sp.diff(Nf[2], y) - sp.diff(Nf[1], z),
                      sp.diff(Nf[0], z) - sp.diff(Nf[2], x),
                      sp.diff(Nf[1], x) - sp.diff(Nf[0], y)])

x, y, z = sp.symbols('x y z')

a1, a2, a3, a4 = sp.symbols('a1 a2 a3 a4')
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
c1, c2, c3, c4 = sp.symbols('c1 c2 c3 c4')
d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')

# L1 = a1 + b1*x + c1*y + d1*z
# L2 = a2 + b2*x + c2*y + d2*z
# L3 = a3 + b3*x + c3*y + d3*z
# L4 = a4 + b4*x + c4*y + d4*z


#L1s, L2s, L3s, L4s = sp.symbols('L1 L2 L3 L4')
#Lss = [L1s, L2s, L3s, L4s]
W12, W13, W14, W23, W24, W34 = sp.symbols('W12 W13 W14 W23 W24 W34')
l12, l13, l14, l23, l24, l34 = sp.symbols('l12 l13 l14 l23 l24 l34')

Wijs = [W12, W13, W14, W23, W24, W34]
lijs = [l12, l13, l14, l23, l24, l34]
ipairs = ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))

#L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')
L1 = sp.Function('L1')(x, y, z)
L2 = sp.Function('L2')(x, y, z)
L3 = sp.Function('L3')(x, y, z)
L4 = sp.Function('L4')(x, y, z)

Ls = [L1, L2, L3, L4]

#########



Order = 0



##########


Nfuncs = []
for (i1, i2), lij, Wij in zip(ipairs, lijs, Wijs):
    Li = Ls[i1-1]
    Lj = Ls[i2-1]

    Wij = Li*grad(Lj) - Lj*grad(Li)

    for n1 in range(Order+1):
        for n2 in range(Order+1):
            for n3 in range(1,Order+2):
                for n4 in range(1,Order+2):
                    other_nums = [num for num in [1,2,3,4] if num not in [i1, i2]]
                    i3, i4 = other_nums
                    nums = [0, 0, 0, 0]
                    nums[i1-1] = n3
                    nums[i2-1] = n4
                    nums[i3-1] = n1
                    nums[i4-1] = n2
                    La = [L1,L2,L3,L4][i3-1]
                    Lb = [L1,L2,L3,L4][i4-1]
                    i, j, k, l = nums
                    beta = n1
                    gamma = n2
                    #print(f'i={i}, j={j}, k={k}, l={l}, beta={beta}, gamma={gamma}')

                    alpha = lij*(Order+2)/(Order+2-gamma-beta)
                    factor = 1
                    if gamma!=0:
                        factor = factor/gamma
                    if beta!=0:
                        factor = factor/beta
                    N = alpha*(Order+2)**2*factor*La*Lb*Phat(L1, i, Order+2)*Phat(L2, j, Order+2)*Phat(L3, k, Order+2)*Phat(L4, l, Order+2)*Wij
                    print(N)
                    Nfuncs.append(N)

V = sp.symbols('V')
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
y1, y2, y3, y4 = sp.symbols('y1 y2 y3 y4')
z1, z2, z3, z4 = sp.symbols('z1 z2 z3 z4')

lam1, lam2, lam3 = sp.symbols('lam1 lam2 lam3')
xf = lam1*x1 + lam2*x2 + lam3*x3 + (1-lam1-lam2-lam3)*x4
yf = lam1*y1 + lam2*y2 + lam3*y3 + (1-lam1-lam2-lam3)*y4
zf = lam1*z1 + lam2*z2 + lam3*z3 + (1-lam1-lam2-lam3)*z4

n = len(Nfuncs)
for i in range(n):
    for j in range(n):
        #print(curl(Ns[i]))
        #print(curl(Ns[j]))
        Nijf = sp.simplify(curl(Nfuncs[i]).dot(Nfuncs[j]))
        #Nijf = Nijf.subs({x: xf, y: yf, z: zf})
        print(f'N{i+1}{j+1} = {Nijf}')
        #Int = 6*V*sp.integrate(sp.integrate(sp.integrate(Nijf, (lam1, 0, 1-lam2-lam3)),(lam2, 0, 1-lam3)), (lam3, 0, 1))
        #print(f'N{i+1}{j+1} = {sp.simplify(Int)}')


