import sympy as sp



def print_basis(name: str, function):
    print('@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8), cache=True, nogil=True)')
    print(f'def {name}(C1, C2, C3, x, y, z, L):')
    print('    a1, b1, c1, d1 = C1')
    print('    a2, b2, c2, d2 = C2')
    print('    a3, b3, c3, d3 = C3')
    print('    fs = np.empty((3, x.shape[0]))')
    f = sp.simplify(function)
    for i in [0,1,2]:
        print(f'    fs[{i},:] = {f[i]}')
    print('    return L*fs')
    print('')
x, y, z = sp.symbols('x y z')


def grad(f):
    return sp.Matrix([sp.diff(f,x), sp.diff(f,y), sp.diff(f,z)])

def curl(A):
    return sp.Matrix([sp.diff(A[2], y) - sp.diff(A[1], z),
                      sp.diff(A[0], z) - sp.diff(A[2], x),
                      sp.diff(A[1], x) - sp.diff(A[0], y)])

a1, b1, c1, d1 = sp.symbols('a1 b1 c1 d1')
a2, b2, c2, d2 = sp.symbols('a2 b2 c2 d2')
a3, b3, c3, d3 = sp.symbols('a3 b3 c3 d3')

L1 = a1 + x*b1 + y*c1 + z*d1
L2 = a2 + x*b2 + y*c2 + z*d2
L3 = a3 + x*b3 + y*c3 + z*d3


N1 = L1*(L2*grad(L1) - L1*grad(L2))
N2 = L2*(L2*grad(L1) - L1*grad(L2))
N3 = L1*L2*grad(L3) - L2*L3*grad(L1)
N4 = L2*L3*grad(L1) - L3*L1*grad(L2)

### FIRST BASIS FUNCTION
print_basis('basis_1', N1)
print_basis('basis_2', N2)
print_basis('basis_3', N3)
print_basis('basis_4', N4)
print_basis('basis_1_curl', curl(N1))
print_basis('basis_2_curl', curl(N2))
print_basis('basis_3_curl', curl(N3))
print_basis('basis_4_curl', curl(N4))