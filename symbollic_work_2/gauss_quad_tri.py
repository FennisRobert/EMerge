import sympy as sp

## Vars

x,y = sp.symbols('xs ys')

## Functions

def vp(func):
    print(func)
    func = sp.simplify(func)
    print('    out=np.empty((2,xs.shape[0]), dtype=np.complex128)')
    print(f'    out[0,:] = {func[0]})')
    print(f'    out[1,:] = {func[1]})')
    print('    return out')

def grad(func):
    return sp.Matrix([sp.diff(func, x), sp.diff(func, y)])


def curl(func):
    ## dfz/y - dfy/z
    ## dfx/z - dfz/x
    ## dfy/x - dfx/y
    return sp.diff(func[1],x) - sp.diff(func[0],y)


#### Prep

a1, a2, a3, b1, b2, b3, c1, c2, c3 = sp.symbols('a1 a2 a3 b1 b2 b3 c1 c2 c3')

L1 = a1 + b1*x + c1*y
L2 = a2 + b2*x + c2*y
L3 = a3 + b3*x + c3*y

Ne1 = L1*(L2*grad(L1) - L1*grad(L2))
Ne2 = L2*(L2*grad(L1) - L1*grad(L2))
Nf1 = L2*(L1*grad(L3) - L3*grad(L1))
Nf2 = L3*(L2*grad(L1) - L1*grad(L2))

Lv = 2*L1*L1 - L1
Le = 4*L1*L2

### Func Terms terms

print('\n FUNCTION \n')

print("Nedelec")
print(f'return {vp(sp.simplify(Ne1))}')
print(f'return {vp(sp.simplify(Ne2))}')
print(f'return {vp(sp.simplify(Nf1))}')
print(f'return {vp(sp.simplify(Nf2))}')
print("Legrange")
print(f'return {sp.simplify(Lv)}')
print(f'return {sp.simplify(Le)}')

## Gradient Terms
print('\n GRADIENT \n')
print("Legrange")
print(f'return {vp(grad(sp.simplify(Lv)))}')
print(f'return {vp(grad(sp.simplify(Le)))}')

### Curl terms

print('\n CURL \n')
print(f'return {sp.simplify(curl(Ne1))}')
print(f'return {sp.simplify(curl(Ne2))}')
print(f'return {sp.simplify(curl(Nf1))}')
print(f'return {sp.simplify(curl(Nf2))}')