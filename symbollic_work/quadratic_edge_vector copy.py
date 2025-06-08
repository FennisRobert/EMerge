from __future__ import annotations
from collections import defaultdict
import re

class Term:

    def __init__(self, symbol: str, vec: bool = False):
        self.s = symbol
        self.vec = vec      
        self._ONE = False
        self._ZERO = False

    @staticmethod
    def ONE() -> Term:
        term = Term('1')
        term._ONE = True
        return term
    
    @staticmethod
    def ZERO() -> Term:
        term = Term('0')
        term._ZERO = True
        return term
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, Term):
            raise TypeError(f'Cannot compare a Term to a {type(other)}')
        return self.s > other.s
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            raise TypeError(f'Can only compare equivalence with another Term object. Not type {type(other)}')
        return (self.s == other.s) and (self._ONE == other._ONE)
    
    def __mul__(self, other) -> Addition:
        return Addition(Product(self)) * other

    def __add__(self, other) -> Addition:
        return Addition(Product(self)) + other

    def __neg__(self) -> Addition:
        return Addition(Product(self, factor=-1))
    
    def __str__(self) -> str:
        return self.s
    
    def __repr__(self):
        return f'Term[{self.__str__()}]'
    
    @property
    def G(self) -> Term:
        # if self.vec:
        #     return Term('∇2'+self.s[1:], vec=False)
        # return Term('∇'+self.s, vec=True)
        if self.vec:
            return Term('G2'+self.s[1:], vec=False)
        return Term('G'+self.s, vec=True)
    
    def dot(self, other) -> Term:
        if not self.vec and other.vec:
            raise ValueError(f'Both terms must be a vector but {self} is {self.vec} and {other} is {other.vec}')
        a = min(self, other)
        b = max(self, other)
        return Term(f'dot({a.s},{b.s})')
    
    def cross(self, other) -> Term:
        if not self.vec and other.vec:
            raise ValueError(f'Both terms must be a vector but {self} is {self.vec} and {other} is {other.vec}')
        if self == other:
            return Term.ZERO(), 1
        a = min(self, other)
        b = max(self, other)
        Q = 1
        if (a==other) and a!=b:
            Q = -1
        return Term(f'{a.s}x{b.s}', vec=True), Q
    
    
class Product:

    def __init__(self, *terms, factor: int = 1):
        self._terms: list[Term] = list(terms)
        self.factor: int = factor

    @property
    def one(self) -> bool:
        return all([t._ONE for t in self._terms]) and self.factor == 1
    
    @property
    def zero(self) -> bool:
        return any([t._ZERO for t in self._terms])
    
    @property
    def terms(self) -> list[Term]:
        return [t for t in self._terms if not t._ONE]
    
    @property
    def vec(self) -> bool:
        return any([t.vec for t in self.terms])

    @property
    def vecterm(self) -> Term:
        for term in self.terms:
            if term.vec:
                return term
        return None
    
    @property
    def nonvec(self) -> Product:
        nonvec = [t for t in self.terms if not t.vec]
        if len(nonvec)==0:
            nonvec = [Term.ONE(),]
        return Product(*nonvec, factor=self.factor)
    
    @property
    def grad_decomposition(self) -> tuple[Product, Term]:
        vec = self.vecterm
        return self.nonvec, vec
    
    def curl(self) -> Addition:
        if not self.vec:
            raise ValueError(f'Can only compute a curl if this is a vector However, this product is not a vector')
        if len(self.terms)==1:
            return Addition()
        else:
            nonvecG = self.nonvec.G
            vec = Addition.from_any(self.vecterm)
            cross = nonvecG.cross(vec)
            return cross

    def __eq__(self, other: Product) -> bool:
        if not isinstance(other, Product):
            raise TypeError(f'Can only compare equivalence with another Product object. Not type {type(other)}')
        if len(self.terms) != len(other.terms):
            return False
        return all([a==b for a,b in zip(self.terms, other.terms)])
    
    def __mul__(self, other) -> Addition:
        return Addition(self) * other
    
    def __add__(self, other) -> Addition:
        return Addition(self) + other
        
    def __neg__(self) -> Addition:
        self.factor *= -1
        return Addition(self)

    def __hash__(self):
        return hash(self.factors())
       
    def __str__(self):
        if self.one:
            return '1'
        if self.zero:
            return '0'
        if self.factor == -1:
            return f'-{"*".join([str(x) for x in self.terms])}'
        if self.factor != 1:
            return f'{self.factor}*{"*".join([str(x) for x in self.terms])}'
        

        return  f'{"*".join([str(x) for x in self.terms])}'
    
    def factors(self):
        return f'{"*".join([str(x) for x in self.terms])}'
    
    def __repr__(self):
        return f'Product[{self.__str__()}]'
    
    def dot(self, other) -> Addition:
        if not self.vec and not other.vec:
            raise ValueError(f'Both products must be vectors but this object {self} is {self.vec} and {other} is {other.vec}')
        nonvec = self.nonvec*other.nonvec
        vecterm = self.vecterm.dot(other.vecterm)
        return nonvec*vecterm

    def cross(self, other) -> Addition:
        if not self.vec and other.vec:
            raise ValueError(f'Both products must be vectors but this object {self} is {self.vec} and {other} is {other.vec}')
        nonvec = (self.nonvec*other.nonvec)
        vecterm, factor = self.vecterm.cross(other.vecterm)
        vector_term = Product(vecterm, factor=factor)
        cross_product = nonvec * vector_term
        return cross_product
    
    @property
    def G(self) -> Addition:
        if not self.vec:
            newterms = []
            for i, tg in enumerate(self.terms):
                others = [x for j, x in enumerate(self.terms) if  i!=j]
                newterms.append(Product(*others, tg.G, factor=self.factor))
            return Addition(*newterms)
        else:
            prods, grad = self.grad_decomposition
            return prods.G.dot(Addition.from_any(grad)) + prods*grad.G
        
class Addition:

    def __init__(self, *products):
        self.terms: list[Product] = list(products)


    def __mul__(self, other) -> Addition:
        mults = [Product(*a.terms, *b.terms, factor=a.factor*b.factor) for a,b in self.permute(Addition.from_any(other))]
        reduced = Addition(*mults).collapse()
        return reduced
    
    def __add__(self, other) -> Addition:
        return Addition(*self.terms, *Addition.from_any(other).terms).collapse()
    
    def __radd__(self, other) -> Addition:
        return self + other
    
    def __sub__(self, other) -> Addition:
        return (self + (-Addition.from_any(other))).collapse()
    
    def permute(self, other: Addition):
        for term1 in self.terms:
            for term2 in other.terms:
                yield term1, term2

    def collapse(self):
        tctr = defaultdict(int)
        new_terms = []

        for prod in self.terms:
            if abs(prod.factor) < 1e-12:
                continue
            if prod.zero:
                continue
            if prod.one:
                continue
            tctr[prod] += prod.factor
        
        for key,value in tctr.items():
            if value == 0:
                continue
            prod = Product(*key._terms, factor=value)
            #key.factor = value
            new_terms.append(prod)
        if not new_terms:
            new_terms = [Product(Term.ONE()),]
        
        #print(f'Turned {self.terms} into {new_terms}')
        self.terms = new_terms
        return self
    
    def curl(self) -> Addition:
        curl_terms = [term.curl() for term in self.terms]
        #print(f' > Curl {self} = {sum(curl_terms).collapse()}')
        return sum(curl_terms).collapse()
    
    def cross(self, other) -> Addition:
        if not isinstance(other,Addition):
            raise TypeError(f'other type must be of type Addition, not {type(other)}')
        cross_terms = [a.cross(b) for a ,b in self.permute(other)]
        #print(f' > {self} x {other} = {sum(cross_terms).collapse()}')
        return sum(cross_terms).collapse()
    
    def dot(self, other) -> Addition:
        if not isinstance(other,Addition):
            raise TypeError(f'other type must be of type Addition, not {type(other)}')
        cross_terms = [a.dot(b) for a ,b in self.permute(other)]
        #print(f' > {self} . {other} = {sum(cross_terms).collapse()}')
        return sum(cross_terms).collapse()
    
    def __neg__(self):
        for term in self.terms:
            term.factor *= -1
        return self.collapse()
    
    def __repr__(self):
        return f'Addition[{self.__str__()}]'
    
    def __str__(self):
        string = '+'.join([str(x) for x in self.terms])
        return string.replace('+-','-').replace('-+','-').replace('++','+').replace('--','+')
    
    @property
    def G(self) -> Addition:
        grad = [t.G for t in self.terms]
        return sum(grad).collapse()
    
    @staticmethod
    def from_any(item):
        if isinstance(item, (float, complex, int)):
            return Addition(Product(Term.ONE(), factor=item))
        if isinstance(item, Term):
            return Addition(Product(item))
        elif isinstance(item, Product):
            return Addition(item)
        elif isinstance(item, Addition):
            return item
        else:
            raise TypeError(f'Cannot turn {item} of type {type(item)} into an Addition')
        return None

def e1f(_A,_B,_C) -> Addition:
    return _A*_B*_A.G - _A*_A*_B.G

def e2f(_A,_B,_C) -> Addition:
    return _B*_B*_A.G - _A*_B*_B.G

def f1f(_A,_B,_C) -> Addition:
    return _A*_B*_C.G - _B*_C*_A.G

def f2f(_A,_B,_C) -> Addition:
    return _B*_C*_A.G - _C*_A*_B.G

A = Term('A')
B = Term('B')
C = Term('C')
D = Term('D')
E = Term('E')
F = Term('F')

tA = Term('tA')
tB = Term('tB')
tC = Term('tC')


pattern = r'([A-Z])\*([A-Z])\*dot\('
replace = r'coeff(V,\1,\2,0,0)*dot('

pattern2 = r'([A-Z])\*([A-Z])\*([A-Z])\*([A-Z])\*dot\('
replace2 = r'coeff(V,\1,\2,\3,\4)*dot('


groups = [('', e1f, '', e1f),
          ('', e1f, '+10', e2f),
          ('+10', e2f, '', e1f),
          ('+10', e2f, '+10', e2f)]

for i1, f1, i2, f2 in groups:
    output = f'            Dmat[ei{i1},ej{i2}] += Li*Lj*CEE*({f1(A,B,E).curl().dot(f2(C,D,F).curl())})'
    print(re.sub(pattern, replace, output))
    output = f'            Fmat[ei{i1},ej{i2}] += Li*Lj*CFEE*({f1(A,B,E).dot(f2(C,D,F))})'
    print(re.sub(pattern2, replace2, output))

print('')
print('')

groups = [('', e1f, '+6', f1f, 'Lac'),
          ('', e1f, '+16', f2f, 'Lab'),
          ('+10', e2f, '+6', f1f, 'Lac'),
          ('+10', e2f, '+16', f2f, 'Lab')]

for i1, f1, i2, f2, lstr in groups:
    output = f'            Dmat[ei{i1},ej{i2}] += Li*{lstr}*CEF*({f1(A,B,E).curl().dot(f2(C,D,F).curl())})'
    print(re.sub(pattern, replace, output))
    output = f'            Fmat[ei{i1},ej{i2}] += Li*{lstr}*CFEF*({f1(A,B,E).dot(f2(C,D,F))})'
    print(re.sub(pattern2, replace2, output))

print('')
print('')

groups = [('+6', f1f,'Lac', '', e1f),
          ('+6', f1f,'Lac', '+10', e2f),
          ('+16', f2f,'Lab', '', e1f),
          ('+16', f2f,'Lab', '+10', e2f)]

for i1, f1, lstr, i2, f2 in groups:
    output = f'            Dmat[ei{i1},ej{i2}] += Lj*{lstr}*CFE*({f1(A,B,E).curl().dot(f2(C,D,F).curl())})'
    print(re.sub(pattern, replace, output))
    output = f'            Fmat[ei{i1},ej{i2}] += Lj*{lstr}*CFFE*({f1(A,B,E).dot(f2(C,D,F))})'
    print(re.sub(pattern2, replace2, output))

print('')
print('')

groups = [('+6', f1f, 'Lac1','+6', f1f,'Lac2'),
          ('+6', f1f, 'Lac1','+16', f2f,'Lab2'),
          ('+16', f2f,'Lab1', '+6', f1f, 'Lac2'),
          ('+16', f2f,'Lab1', '+16', f2f,'Lab2')]

for i1, f1,lstr1,i2, f2,lstr2 in groups:
    output = f'            Dmat[ei{i1},ej{i2}] += {lstr1}*{lstr2}*CFF*({f1(A,B,E).curl().dot(f2(C,D,F).curl())})'
    print(re.sub(pattern, replace, output))
    output = f'            Fmat[ei{i1},ej{i2}] += {lstr1}*{lstr2}*CFFF*({f1(A,B,E).dot(f2(C,D,F))})'
    print(re.sub(pattern2, replace2, output))


pattern3 = r'([tA-Z]+)\*([tA-Z]+)\*([tA-Z]+)\*([tA-Z]+)\*dot\('
replace3 = r'area_coeff(Area,\1,\2,\3,\4)*dot('

########### SURFACE Integral

#def e1f(_A,_B,_C) -> Addition:
#    return _A*_B.G - _B*_A.G#_A*_B*_A.G - _A*_A*_B.G


#def e2f(_A,_B,_C) -> Addition:
#    return _B*_B*_A.G - _A*_B*_B.G

groups = [('', e1f, '', e1f),
          ('', e1f, '+4', e2f),
          ('+4', e2f, '', e1f),
          ('+4', e2f, '+4', e2f)]

for i1, f1, i2, f2 in groups:
    output = f'            Bmat[ei{i1},ej{i2}] += Li*Lj*COEFF*({f1(A,B,E).dot(f2(C,D,F))})'
    print(re.sub(pattern3, replace3, output))
print('')
print('')

groups = [('', e1f),
          ('+4', e2f)]
          

for i1, f1 in groups:
    output2 = f'            Bmat[ei{i1},3] += Li*Lt1*COEFF*({f1(A,B,None).dot(f1f(tA,tB,tC))})'
    output3 = f'            Bmat[ei{i1},7] += Li*Lt2*COEFF*({f1(A,B,None).dot(f2f(tA,tB,tC))})'
    output4 = f'            Bmat[3,ei{i1}] += Lt1*Li*COEFF*({f1f(tA,tB,tC).dot(f1(A,B,None))})'
    output5 = f'            Bmat[7,ei{i1}] += Lt2*Li*COEFF*({f2f(tA,tB,tC).dot(f1(A,B,None))})'
    print(output2)
    print(output3)
    print(output4)
    print(output5)

print('')
print('')

print(re.sub(pattern3, replace3, f'            Bmat[3,3] += Lt1*Lt1*COEFF*({f1f(tA,tB,tC).dot(f1f(tA,tB,tC))})'))
print(re.sub(pattern3, replace3, f'            Bmat[3,7] += Lt1*Lt2*COEFF*({f1f(tA,tB,tC).dot(f2f(tA,tB,tC))})'))
print(re.sub(pattern3, replace3, f'            Bmat[7,3] += Lt2*Lt1*COEFF*({f2f(tA,tB,tC).dot(f1f(tA,tB,tC))})'))
print(re.sub(pattern3, replace3, f'            Bmat[7,7] += Lt2*Lt2*COEFF*({f2f(tA,tB,tC).dot(f2f(tA,tB,tC))})'))