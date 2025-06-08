from __future__ import annotations
from collections import defaultdict
from typing import Callable
import re
from loguru import logger

class Translator:

    def __init__(self):
        self.multsym = '*'
        self.gradstr = '∇'
        self.curlstr = '∇x'
        self.divstr = '∇.'
        self.nablastr = '∇2'
        self.gradtanstr = '∇t'
        self.nablatanstr = '∇2t'
        self.dotstr = lambda a,b: f'dot({a.s},{b.s})'
        self.crossstr = lambda a,b: f'({a.s}x{b.s})'
        self.matmulfunc = lambda a,b: f'matmul({a},{b})'

TRANSLATOR = Translator()

funccallpat = r'([a-z]+\([\@a-zA-Z0-9_]+,[\@a-zA-Z0-9_]+\))'

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def pair_iter(iter1, iter2):
    for i1 in iter1:
        for i2 in iter2:
            yield i1, i2

syms = ['*','+','-']

def coeff_fun(string: str, fname: str):
    pre = string[0]
    post = string[-1]
    if pre not in syms:
        pre = ''
    if post not in syms:
        post = ''
    letters = string.split('*')
    letters = [c[-1] for c in letters if c]
    letters = letters + ['0',]*(4-len(letters))
    return pre + f'{fname}[' +','.join(letters) + ']' + post

def coeff_var(string: str, varname: str):
    pre = string[0]
    post = string[-1]
    letters = string.split('*')
    return pre + f'{varname}' + ''.join(letters) + post


class ConstantSet:

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.constants = {}
        self.LCTR = 0
        self.NCTR = 1
    
    def new(self) -> str:
        self.LCTR += 1
        if self.LCTR > len(LETTERS)-1:
            self.LCTR = 0
            self.NCTR += 1
        
        return self.prefix + LETTERS[self.LCTR]+str(self.NCTR)


def extract_defs(lines: list[str], definitions: list[str], tokens: ConstantSet) -> list[str]:
    calls = defaultdict(int)
    ctr = 0
    for line in lines:
        for match in re.finditer(funccallpat, line):
            calls[match.group(0)] += 1
    
    translations = []

    for call in calls:
        callrep = tokens.new()#call.replace('(','').replace(')','').replace(',','').replace('@','')
        translations.append((call, callrep))
        if calls[call] > 0:
            ctr += 1
            definitions.append(f'{callrep} = {call}')
    
    newlines = []
    for line in lines:
        newline = line
        for call, callrep in translations:
            newline = newline.replace(call, callrep)
        newlines.append(newline)

    if ctr > 0:
        newlines, definitions = extract_defs(newlines, definitions, tokens)
    return newlines, definitions

class Term:

    def __init__(self, symbol: str, 
                 vec: bool = False,
                 mat: bool = False):
        self.s = symbol
        self.vec = vec      
        self.mat = mat
        self._ONE = False
        self._ZERO = False

    @staticmethod
    def ONE() -> Term:
        term = Term('1')
        term._ONE = True
        return term
    
    @staticmethod
    def ZERO(vec=False) -> Term:
        term = Term('0', vec=vec)
        term._ZERO = True
        return term
    
    @property
    def asadd(self) -> Addition:
        return Addition(Product(self))
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, Term):
            raise TypeError(f'Cannot compare a Term to a {type(other)}')
        return self.s > other.s
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            raise TypeError(f'Can only compare equivalence with another Term object. Not type {type(other)}')
        return (self.s == other.s) and (self._ONE == other._ONE)
    
    def __mul__(self, other) -> Addition:
        return self.asadd * other

    def __rmul__(self, other) -> Addition:
        return self.asadd * other
    
    def __add__(self, other) -> Addition:
        return self.asadd + other

    def __neg__(self) -> Addition:
        return -self.asadd
    
    def __str__(self) -> str:
        return self.s
    
    def __repr__(self):
        return f'Term[{self.__str__()}]'
    
    @property
    def G(self) -> Term:
        if self.vec:
            return Term(TRANSLATOR.nablastr+self.s[1:], vec=False)
        return Term(TRANSLATOR.gradstr+self.s, vec=True)
    
    @property
    def Gt(self) -> Term:
        return Term(TRANSLATOR.gradtanstr+self.s, vec=True)
    
    def matscale(self, mat: str) -> Term:
        if not self.vec:
            raise ValueError(f'Term {self} is not a vector term')
        return Term(TRANSLATOR.matmulfunc(mat, self.s), vec=self.vec)
    
    def dot(self, other) -> Term:
        if not self.vec and other.vec:
            raise ValueError(f'Both terms must be a vector but {self} is {self.vec} and {other} is {other.vec}')
        if self._ZERO or other._ZERO:
            return Term.ZERO()
        a = min(self, other)
        b = max(self, other)
        return Term(TRANSLATOR.dotstr(a,b), vec=False)
    
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
        return Term(TRANSLATOR.crossstr(a,b), vec=True), Q
    
    @staticmethod
    def terms(names: str) -> list[Term]:
        """
        Create a list of Term objects from a string of names separated by commas.
        """
        terms = [Term(name.strip()) for name in names.split(' ')]
        return terms
    
class Product:

    def __init__(self, *terms, factor: int = 1):
        self._terms: list[Term] = list(terms)
        self.factor: int = factor

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
        if self.constant:
            return str(self.factor)
        if self.zero:
            return '0'
        if self.factor == -1:
            return f'-{"*".join([str(x) for x in self.terms])}'
        if self.factor != 1:
            return f'{self.factor}*{"*".join([str(x) for x in self.terms])}'
        

        return  f'{"*".join([str(x) for x in self.terms])}'

    @property
    def one(self) -> bool:
        '''Is True if this product is equal to 1'''
        return all([t._ONE for t in self._terms]) and self.factor == 1
    
    @property
    def zero(self) -> bool:
        '''Is True if this product is equal to 0'''
        return any([t._ZERO for t in self._terms])
    
    @property
    def constant(self) -> bool:
        '''Is True if this product is a constant'''
        return all([t._ONE for t in self._terms])
    
    @property
    def terms(self) -> list[Term]:
        return sorted([t for t in self._terms if not t._ONE])
    
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
    
    def matscale(self, mat: str) -> Product:
        if not self.vec:
            raise ValueError(f'Term {self} is not a vector term')
        self._terms = self.nonvec._terms + [self.vecterm.matscale(mat),]
    
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


    def factors(self):
        return f'{"*".join([str(x) for x in self.terms])}'
    
    def __repr__(self):
        return f'Product[{self.__str__()}]'
    
    def dot(self, other) -> Addition:
        if self.zero or other.zero:
            return Addition.from_any(Term.ZERO())
        if not self.vec and not other.vec:
            raise ValueError(f'Both products must be vectors but this object {self} is {self.vec} and {other} is {other.vec}')
        nonvec = self.nonvec*other.nonvec
        vecterm = self.vecterm.dot(other.vecterm)
        return nonvec*vecterm

    def cross(self, other) -> Addition:
        ''' Implements the identity: fa x gb = f*g*(a x b)'''
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
    
    @property
    def Gt(self) -> Addition:
        if not self.vec:
            newterms = []
            for i, tg in enumerate(self.terms):
                others = [x for j, x in enumerate(self.terms) if  i!=j]
                newterms.append(Product(*others, tg.Gt, factor=self.factor))
            return Addition(*newterms)
        else:
            prods, grad = self.grad_decomposition
            return prods.Gt.dot(Addition.from_any(grad)) + prods*grad.Gt
        
class Addition:

    def __init__(self, *products):
        self.terms: list[Product] = list(products)


    def __mul__(self, other) -> Addition:
        mults = [Product(*a.terms, *b.terms, factor=a.factor*b.factor) for a,b in self.permute(Addition.from_any(other))]
        reduced = Addition(*mults).collapse()
        return reduced
    
    def __add__(self, other) -> Addition:
        #logger.debug(f'Adding {self} + {other}')    
        return Addition(*self.terms, *Addition.from_any(other).terms).collapse()
    
    def __radd__(self, other) -> Addition:
        return self + other
    
    def __sub__(self, other) -> Addition:
        return (self + (-Addition.from_any(other))).collapse()
    
    def __neg__(self):
        for term in self.terms:
            term.factor *= -1
        return self.collapse()
    
    def __repr__(self):
        return f'Addition[{self.__str__()}]'
    
    def __str__(self):
        string = '+'.join([str(x) for x in self.terms])
        return string.replace('+-','-').replace('-+','-').replace('++','+').replace('--','+')
    
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
            tctr[prod] += prod.factor
        
        for key,value in tctr.items():
            if value == 0:
                continue
            prod = Product(*key._terms, factor=value)
            #key.factor = value
            new_terms.append(prod)

        if not new_terms:
            new_terms = [Product(Term.ZERO()),]
        
        #print(f'Turned {self.terms} into {new_terms}')
        self.terms = new_terms
        return self
    
    def matscale(self, mat: str) -> Addition:
        for term in self.terms:
            if term.vec:
                term.matscale(mat)
        return self

    def curl(self) -> Addition:

        curl_terms = [term.curl() for term in self.terms]

        return sum(curl_terms).collapse()
    
    def cross(self, other) -> Addition:
        if not isinstance(other,Addition):
            raise TypeError(f'other type must be of type Addition, not {type(other)}')
        cross_terms = [a.cross(b) for a ,b in self.permute(other)]
        return sum(cross_terms).collapse()
    
    def dot(self, other) -> Addition:
        if not isinstance(other,Addition):
            raise TypeError(f'other type must be of type Addition, not {type(other)}')
        cross_terms = [a.dot(b) for a ,b in self.permute(other)]
        return sum(cross_terms).collapse()
    
    @property
    def G(self) -> Addition:
        grad = [t.G for t in self.terms]
        return sum(grad).collapse()
    
    @property
    def Gt(self) -> Addition:
        grad = [t.Gt for t in self.terms]
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
    

class FunctionSet:

    def __init__(self):
        self.modes: list[tuple[str, Callable]] = []
        self.N: int = -1
    
    @property
    def nmodes(self) -> int:
        return len(self.modes)

    @property
    def nT(self) -> int:
        return self.N*self.nmodes
    
    def add(self, mode: Callable, constant: str = ''):
        """
        Add a function to the set of functions.
        """
        if not callable(mode):
            raise TypeError(f'Mode must be callable, not {type(mode)}')
        self.modes.append((constant, mode))

    def iterate(self):
        """
        Iterate over the modes in the set.
        """
        if not self.modes:
            yield None, (None, None)
        for mode in self.modes:
            yield self.nmodes, mode

    def getmode(self, N: int) -> tuple[int, Callable]:
        """
        Get the Nth mode from the set.
        """
        if N >= len(self.modes):
            return None, (None, None)
        return self.N, self.modes[N]
    
class ElementSet:

    def __init__(self, tet: bool = False, tri: bool = False):
        self.vmodes: FunctionSet = FunctionSet()
        self.emodes: FunctionSet = FunctionSet()
        self.fmodes: FunctionSet = FunctionSet()

        if tet:
            self.vmodes.N = 4
            self.emodes.N = 6
            self.fmodes.N = 4
        if tri:
            self.vmodes.N = 3
            self.emodes.N = 3
            self.fmodes.N = 1

        self.termnamesA = 'A₁ B₁ C₁'
        self.termnamesB = 'A₂ B₂ C₂'
        self.coefffunction = 'AREA_COEFF'
        self.coeff_letter = 'A'

    def generate_modeset(self):
        maxMode = max((self.vmodes.nmodes, self.emodes.nmodes, self.fmodes.nmodes))
        
        all_modes = {
            'V': [],
            'E': [],
            'F': []
        }

        N = 0
        for i in range(maxMode):
            
            n, (c,f) = self.vmodes.getmode(i)
            if f is not None:
                all_modes['V'].append((i, N, f, c))
                N += n
            n, (c,f) = self.emodes.getmode(i)
            if f is not None:
                all_modes['E'].append((i, N, f, c))
                N += n
            n, (c,f) = self.fmodes.getmode(i)
            if f is not None:
                all_modes['F'].append((i, N, f, c))
                N += n
        return all_modes
    
    def generate_basis(self, other: ElementSet, 
                       var: str, 
                       constant_prefix: str, 
                       variational: Callable, 
                       const: str = '', 
                       Nspace: int = 1,
                       rename: bool = True,
                       int_fun: bool = False,
                       start_index: int = 0):
        """
        Generate the basis functions for the element set.
        """

        modes1 = self.generate_modeset()
        modes2 = other.generate_modeset()
        
        
        pattern = r'[+*-]?\b(?:[A-F]+(?:\*)+)+\b'

        A1, B1, C1 = Term.terms(self.termnamesA)
        A2, B2, C2 = Term.terms(self.termnamesB)

        for tA, tB in pair_iter(['V', 'E', 'F'], ['V', 'E', 'F']):
            mA = modes1[tA]
            mB = modes2[tB]

            defs = []
            lines = []

            for (modeNr1, Nctr1, func1, const1), (modeNr2, Nctr2, func2, const2)  in pair_iter(mA, mB):
                
                c1 = const1
                if c1:
                    c1 = c1 + '1*'
                c2 = const2
                if c2:
                    c2 = c2 + '2*'
                string = f'{var}[ei+{start_index+Nctr1},ej+{start_index+Nctr2}] = {const}*{c1}{c2}({variational(func1(A1,B1,C1), func2(A2,B2,C2))})'
                
                replacements = []
                
                mtch = re.findall(pattern, string)
                for mtchstring in mtch:
                    if int_fun:
                        out = coeff_fun(mtchstring, self.coefffunction)
                    else:
                        out = coeff_var(mtchstring, self.coeff_letter)
                        defs.append(f'{out} = {coeff_fun(mtchstring, self.coefffunction)}')
                    replacements.append((mtchstring, out))
                    #defs.append(f'{re.sub(pattern, rep_var, mtchstring)[:-1]} = {re.sub(pattern, rep_int, mtchstring)[:-1]}')
                
                #print('In: ', string)
                replacements = sorted(replacements, key=lambda x: len(x[1]), reverse=True)
                if rename:
                    for a,b in replacements:
                        string = string.replace(a,b)
                #print('Out: ', string)
                #string = re.sub(pattern2, replace2, string)
                #string = re.sub(pattern, replace, string)

                lines.append(string)
        
            if rename:
                lines, defs = extract_defs(lines, defs, ConstantSet(constant_prefix))

            
                for df in defs:
                    print(Nspace*'    ' + df.replace('*',''))
                print('')
            for line in lines:
                print(Nspace*'    ' + line.replace('*',TRANSLATOR.multsym))
            print('')