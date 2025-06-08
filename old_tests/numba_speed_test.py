from numba import njit, c16, f8, i8
import numpy as np
import time

def timeit(func, *args, **kwargs):
    """
    Calls a Numba-compiled function once and prints the execution time.
    
    Parameters:
    - func: A Numba-compiled function
    - *args, **kwargs: Arguments to pass to the function
    """
    # Warm-up to trigger JIT compilation (first call is typically slower)
    #func(*args, **kwargs)
    
    # Precise timing of actual run
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    
    print(f"Execution time: {(end - start) * 1e6:.2f} µs")
    return result

### Complex conversion

N = 10_000

@njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def convert_v1(a, b, num):
    out = np.empty(a.shape[0], dtype=np.complex128)
    for i in range(num):
        c = a[i]**2 + b[i]+4 + 0*1j
        out[i] = 2*c
    return out

njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def convert_v2(a, b, num):
    out = np.empty(a.shape[0], dtype=np.complex128)
    for i in range(num):
        c = np.complex128(a[i]**2 + b[i]+4)
        out[i] = 2*c
    return out

njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def convert_v3(a, b, num):
    out = np.empty(a.shape[0], dtype=np.complex128)
    for i in range(num):
        c = np.astype(a[i]**2 + b[i]+4, np.complex128)
        out[i] = 2*c
    return out


a = np.random.rand(N)
b = np.random.rand(N)

timeit(convert_v1, a,b,N)
timeit(convert_v2, a,b,N)
timeit(convert_v3, a,b,N)

### FUnction calling

N = 1_000
a = np.random.rand(N)
b = np.random.rand(N)

@njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def subfun1(a, b, num):
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        c = a[i]**2 + b[i]+4 + N
        out[i] = 2*c
    return out + 0*1j


@njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function1(a, b, num):
    out = np.empty(a.shape[0], dtype=np.complex128)
    for i in range(num):
        out[i] = np.sum(subfun1(a,b,i))
    return out

@njit(c16[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function2(a, b, num):
    out = np.empty(a.shape[0], dtype=np.complex128)
    out_2 = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        for j in range(num):
            c = a[j]**2 + b[j]+4 + N
            out_2[j] = 2*c
        out[i] = np.sum(out_2)+0*1j
    return out

print('Nested function call \n')
print('Function in function')
timeit(function1, a,b,N)
print('Direct')
timeit(function2, a,b,N)
        
###### Nested call 2

N = 1_000_000
a = np.random.rand(N)
b = np.random.rand(N)

@njit(f8(f8, f8), fastmath=True, cache=True, nogil=True)
def subfun2(a, b):
    return a+b


@njit(f8[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function1_2(a, b, num):
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        out[i] = subfun2(a[i], b[i])
    return out

@njit(f8[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function2_2(a, b, num):
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        out[i] = a[i]+b[i]
    return out

print('Nested function version 2 \n')
print('Function in Function')
timeit(function1_2, a,b,N)
print('Direct')
timeit(function2_2, a,b,N)



###### Nested arguments

N = 10_000_000
a = np.random.rand(N)
b = np.random.rand(N)

@njit(f8(f8), fastmath=True, cache=True, nogil=True)
def operation(a):
    return 2*a + a/3 -a*6

@njit(f8(f8, f8), fastmath=True, cache=True, nogil=True)
def subfun3(a, b):
    return 5*a+3*b

@njit(f8[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function1_3(a, b, num):
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        out[i] = subfun3(operation(a[i]), operation(b[i]))
    return out

@njit(f8[:](f8[:], f8[:], i8), fastmath=True, cache=True, nogil=True)
def function2_3(a, b, num):
    out = np.empty(a.shape[0], dtype=np.float64)
    for i in range(num):
        num1 = operation(a[i])
        num2 = operation(b[i])
        out[i] = subfun3(num1, num2)
    return out

print('Nested function version 2 \n')
print('Function in Function')
timeit(function1_3, a,b,N)
print('First write to local value')
timeit(function2_3, a,b,N)