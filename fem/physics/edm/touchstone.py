import os
from typing import Literal
import numpy as np
import skrf as rf

def generate_touchstone(filename: str,
                        freq: np.ndarray,
                        Smat: np.ndarray,
                        data_format: Literal['RI','MA','DB']) -> None:
    """
    Export S-parameter data to a Touchstone file.

    Parameters
    ----------
    filename : str
        Base filename (may include path). If no extension is given, one
        will be added of the form '.sNp' where N = number of ports.
    freq : np.ndarray
        1-D array of M frequency points, in Hz.
    Smat : np.ndarray
        3-D array of S-parameters with shape (N, N, M).
    data_format : {'RI','MA','DB'}
        RI = real&imag, MA = magnitude&angle (deg), DB = dB&angle (deg).
    """
    # --- validations ---
    if Smat.ndim != 3:
        raise ValueError(f"Smat must be 3-D with shape (N,N,M), got ndim={Smat.ndim}")
    M, N1, N2 = Smat.shape
    if N1 != N2:
        raise ValueError(f"Smat must be square in its first two dims, got {N1}×{N2}")
    if freq.ndim != 1 or freq.size != M:
        raise ValueError(f"freq must be 1-D of length {M}, got shape {freq.shape}")

    # --- build scikit-rf objects ---
    # convert freq array (Hz) into an skrf Frequency object
    freq_obj = rf.Frequency.from_f(freq, unit='Hz')
    # build the Network
    ntwk = rf.Network(frequency=freq_obj, s=Smat)

    # --- determine output filename & extension ---
    base, ext = os.path.splitext(filename)
    if ext == '':
        ext = f".s{N1}p"
    filename_out = base + ext

    # --- write the Touchstone file ---
    # skrf expects the format keyword lowercase: 'ri', 'ma', or 'db'
    ntwk.write_touchstone(filename_out, form=data_format.lower())
    print(f"T ouchstone file written to '{filename_out}'")

def generate_touchstone_NEW(filename: str,
                        freq: np.ndarray,
                        Smat: np.ndarray,
                        data_format: Literal['RI','MA','DB']):
    lines = []
    SA = Smat.real
    SB = Smat.imag

    if data_format=='MA':
        SA, SB = np.sqrt(SA**2 + SB**2), np.atan2(SB,SA)*180/np.pi
    elif data_format=='DB':
        SA, SB = 20*np.log10(np.sqrt(SA**2 + SB**2)), np.atan2(SB,SA)*180/np.pi
    
    options_line = f'# GHZ S {data_format} R 50'