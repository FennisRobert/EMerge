import sympy as sp

def replace(string):
    mapping = 'abcdefghi'
    out = [f's[{i},{j}]' for i in range(3) for j in range(3)]
    for m,o in zip(mapping, out):
        string = string.replace(m,o)
    return string
a, b, c, d, e, f, g, h, i = sp.symbols('a b c d e f g h i')

M = sp.Matrix([[a,b,c],[d,e,f],[g,h,i]])

Mi = sp.inv_quick(M)

Mi = Mi*sp.det_quick(M)
print(f'    det = {replace(str(sp.det_quick(M)))}')
for i in range(3):
    for j in range(3):
        print(f'    out[{i},{j}] = {replace(str(Mi[i,j]))}')
print(f'    out = out*det')
#print(Mi*sp.det_quick(M))
