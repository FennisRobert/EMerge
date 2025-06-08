import numpy as np

def norm(Field: np.ndarray) -> np.ndarray:
    ''' Computes the complex norm of a field (3,N)'''
    return np.sqrt(np.abs(Field[0,:])**2 + np.abs(Field[1,:])**2 + np.abs(Field[2,:])**2)
