from basis import Term, TRANSLATOR, ElementSet


TRANSLATOR.gradstr = 'G'
TRANSLATOR.gradtanstr = 'Gt'

TRANSLATOR.dotstr = lambda a, b: f'dot_c({a},{b})'
TRANSLATOR.crossstr = lambda a, b: f'cross_c({a},{b})'


Legrange2 = ElementSet(tri=True)
Legrange2.termnamesA = 'A B E'
Legrange2.termnamesB = 'C D F'

Legrange2.vmodes.add(lambda A, B, C: (2*A-1)*A)
Legrange2.emodes.add(lambda A, B, C: 4*A*B)

#Legrange2.generate_basis('X',lambda a,b: a.G.dot(b.G))

M = Term('M', mat=True)

Li = Term('Li')
Lj = Term('Lj')
Lab = Term('Lab')
Lac = Term('Lac')

def e1f(_A,_B,_C):
    return _A*_B*_A.G - _A*_A*_B.G

def e2f(_A,_B,_C):
    return _B*_B*_A.G - _A*_B*_B.G

def f1f(_A,_B,_C):
    return _A*_B*_C.G - _B*_C*_A.G

def f2f(_A,_B,_C):
    return _B*_C*_A.G - _C*_A*_B.G

Nedelec2 = ElementSet(tet=True)
Nedelec2.emodes.add(e1f, 'L')
Nedelec2.emodes.add(e2f, 'L')
Nedelec2.fmodes.add(f1f, 'Lac')
Nedelec2.fmodes.add(f2f, 'Lab')

Nedelec2.termnamesA = 'A B E'
Nedelec2.termnamesB = 'C D F'
Nedelec2.generate_basis('Dmat', 'A', lambda a,b: a.curl().dot(b.curl().matscale('Ms')), const='KA', Nspace=0, rename=False)

print('Mass')
Nedelec2.generate_basis('Fmat', 'B', lambda a,b: a.dot(b.matscale('Mm')), const='KB', Nspace=0, rename=False)