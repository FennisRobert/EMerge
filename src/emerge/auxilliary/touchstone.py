# EMerge is an open source Python based FEM EM simulation module.
# Copyright (C) 2025  Robert Fennis.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, see
# <https://www.gnu.org/licenses/>.


# Last Cleanup: 2025-01-01
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Callable
import re
from loguru import logger
from scipy.interpolate import interp1d
from emsutil import Saveable

_FUNIT = {
    'hz': 1,
    'khz': 1000,
    'mhz': 1e6,
    'ghz': 1e9,
}
_NPORTS = {
    '.s1p': 1,
    '.s2p': 2,
    '.s3p': 3,
    '.s4p': 4,
    '.snp': None
}


############################################################
#                         FUNCTIONS                        #
############################################################


def _ma_ri(mag: float, angle: float) -> complex:
    """Converts magnitude and angle to complex number.

    Args:
        mag (float): The magnitude of the complex number.
        angle (float): The angle of the complex number.

    Returns:
        complex: The complex number.
    """    
    return mag * np.exp(1j * np.radians(angle))

def _db_ri(db: float, angle: float) -> complex:
    """Converts dB and angle to complex number.

    Args:
        db (float): The magnitude in dB.
        angle (float): The angle in degrees.

    Returns:
        complex: The complex number.
    """    
    return 10**(db/20) * np.exp(1j * np.radians(angle))

def _ri_ri(real: float, imag: float) -> complex:
    """Converts real and imaginary parts to complex number.

    Args:
        real (float): The real part of the complex number.
        imag (float): The imaginary part of the complex number.

    Returns:
        complex: The complex number.
    """    
    return real + 1j * imag

_DATAMAP = {
    'ma': _ma_ri,
    'db': _db_ri,
    'ri': _ri_ri
}


############################################################
#                    EQUIVALENT CIRCUIT                   #
############################################################


class Zfmodel(Saveable):
    def __init__(self, fs: np.ndarray, zmat: np.ndarray):
        self.fs: np.ndarray = fs
        self.zmat: np.ndarray = zmat.flatten()

    def __call__(self, f):
        from scipy.interpolate import interp1d

        return interp1d(self.fs, self.zmat, kind='cubic')(f)
    
    

############################################################
#                        MAIN CLASS                       #
############################################################


class TouchstoneData:
    """A touchstone file importer for S-parameter data."

    The TouchstoneData class can be used to quickly import Touchstone files into your model to plot the data in your graphs.
    
    Args:
        filename (str): The filename of the touchstone file.
        n_ports (int, optional): The number of ports. Defaults to None.
        ignore_extension (bool, optional): Ignore the file extension. Defaults to False.
        interp_type (str, optional): The interpolation type. Defaults to 'cubic'.

    Example:
        ```python
        touchstone = ToustoneData('touchstone.s2p')
        touchstone.renormalize(75)
        ```

    """
    def __init__(self, filename: str,
                 n_ports: int = None,
                 ignore_extension: bool = False,
                 interp_type: str = 'cubic'):
        super().__init__()
        path = Path(filename)
        if not path.exists():
            raise ValueError(f'The provided filename {path} does not exist.')

        if path.suffix not in ('.s1p','.s2p','.s3p','.s4p','.snp','.ts') and not ignore_extension:
            raise ValueError(f'Can only open touchstone files, files with {path.suffix} are not a valid touchstone extension.')
        
        with open(filename,'r') as f:
            self._lines = f.read().split('\n')
        
        self.Sdata = None

        self.freq_unit: float = 1e9
        self.param_type: str = 's'
        self.data_type: str = 'ma'
        self._converter: Callable = _DATAMAP['ma']
        self.refimp: float = 50
        self.n_ports: int = None
        self.interp_type: str = interp_type
        self.Sdata: np.ndarray = None
        self.Fdata: np.ndarray = None

        if self.param_type != 's':
            raise ValueError('Only S-parameter touchstone files are supported as of this moment.')

        self._parse_touchstone()
    
    @property 
    def f(self) -> np.ndarray:
        return self.Fdata
    
    def summarize_data(self):
        # ALLOWED PRINT
        print(f'Frequency unit: {self.freq_unit}')
        # ALLOWED PRINT
        print(f'Parameter type: {self.param_type}')
        # ALLOWED PRINT
        print(f'Data type: {self.data_type}')
        # ALLOWED PRINT
        print(f'Converter: {self._converter}')
        # ALLOWED PRINT
        print(f'Reference impedance: {self.refimp}')
        # ALLOWED PRINT
        print(f'Number of ports: {self.n_ports}')

    def _parse_touchstone(self) -> None:
        s_data = []
        f_data = []
        s_collector = []
        freq = None
        for line in self._lines:
            if not line:
                continue
            
            stripped = line.strip().lower()
            # Comment line
            if stripped[0]=='!':
                continue
            
            # Options line
            if stripped[0]=='#':
                funit = re.findall(r'(hz|khz|mhz|ghz)', stripped)
                param_type = re.findall(r'([syzgh])', stripped)
                data_type = re.findall(r'(ma|db|ri)', stripped)
                refimp = re.findall(r'r\s+(\d+)', stripped)

                if funit:
                    self.freq_unit = _FUNIT[funit[0]]
                if param_type:
                    self.param_type = param_type[0]
                if data_type:
                    self.data_type = data_type[0]
                    self._converter = _DATAMAP[data_type[0]]
                if refimp:
                    self.refimp = float(refimp[0])
            
                continue
            
            # Version 2.0 Keyword line:
            if stripped[0]=='[':
                logger.warning('Version 2.0 touchstone files are not supported yet and keywords will be ignored')
                continue
            # Data line
            # remove the exclamation mark plus all symbols behind it if it occurs
            line = line.split('!')[0]

            nums = [float(x) for x in line.split(' ') if x]

            if len(nums)%2 == 1:
                if len(s_collector) > 0:
                    s_data.append(s_collector)
                    f_data.append(freq)
                s_collector = nums[1:]
                freq = nums[0]
            else:
                s_collector.extend(nums)
            
        s_data.append(s_collector)
        f_data.append(freq)

        ssample = s_data[0]
        self.n_ports = int(np.sqrt(len(ssample) // 2))
        self.n_nodes = self.n_ports

        self.n_frequencies = len(f_data)
        self.Fdata = np.array(f_data) * self.freq_unit
        self.Sdata = np.zeros((self.n_frequencies, self.n_ports, self.n_ports), dtype=complex)

        for i, s in enumerate(s_data):
            sdata = [self._converter(s1,s2) for s1,s2 in zip(s[::2], s[1::2])]
            Smat = np.array(sdata).reshape(self.n_ports, self.n_ports)
            if self.n_ports == 2:
                Smat = Smat.T
            self.Sdata[i,:,:] = Smat

    def renormalize(self, new_impedance: float) -> TouchstoneData:
        """ Renormalize the touchstone file to a new impedance.

        Args:
            new_impedance (float): The new impedance to normalize the touchstone file to.

        Returns:
            FileBasedNPort: The renormalized touchstone file subcircuit.
        """        
        Zold = self.refimp
        Znew = new_impedance
        A = np.eye(self.n_ports, self.n_ports) * np.sqrt(Znew/Zold)*(1/(Znew+Zold))
        R = np.eye(self.n_ports, self.n_ports) * ((Znew-Zold)/(Znew+Zold))
        iA = np.linalg.pinv(A)
        E = np.eye(self.n_ports, self.n_ports)
        for iif in range(self.n_frequencies):
            S = self.Sdata[iif,:,:]
            self.Sdata[iif,:,:] = iA @ (S - R) @ np.linalg.pinv(E - R @ S) @ A
        self.refimp = new_impedance
        return self
    
    def is_reciprocal(self, tolerance: float = 1e-6) -> bool:
        """Returns True if the S-parameters are that of a reciprocal component up to a tolerance precision.

        Args:
            tolerance (float, optional): _description_. Defaults to 1e-6.

        Returns:
            bool: _description_
        """
        if self.n_ports == 1:
            return True
        for i in range(1, self.n_ports + 1):
            for j in range(i+1, self.n_ports + 1):
                if np.any(np.abs(self.S(i,j)-self.S(j,i)) > tolerance):
                    return False
        return True

    def is_passive(self) -> bool:
        """Returns True if the S-parameters of are that of a passive component (All S-parameters smaller than 1.0 in magnitude)

        Returns:
            bool: _description_
        """
        return np.all(np.abs(self.Sdata.flatten()) <= 1.0)
    
    def is_singular(self):
        """
        Checks for singularity and branches between Z-conversion 
        and ABCD-parameter extraction for series components.
        """
        S = self.Sdata
        
        # Check for singularity using the condition number
        # A high condition number (e.g., > 1e12) indicates near-singularity
        I = np.eye(2)[np.newaxis, :, :]
        I_minus_S = I - S
        cond_nums = np.linalg.cond(I_minus_S)
        
        # Threshold for deciding if we should use the Z-conversion or the series model
        # If the matrix is singular, we treat it as a series element
        return np.any(cond_nums > 1e12)
    

    def z_shunt_t(self) -> Zfmodel:
        """Returns the shunt impedance for a 2-port equivalent T-network

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        return Zfmodel(self.Fdata, zmat[:,0,1])

    def z_series_t_1(self) -> Zfmodel:
        """Returns the series impedance for a 2-port equivalent T-network on port 1.

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        z11 = zmat[:,0,0]
        z12 = zmat[:,1,0]
        z22 = zmat[:,1,1]
        return Zfmodel(self.Fdata, z11-z12)
    
    def z_series_t_2(self) -> Zfmodel:
        """Returns the series impedance for a 2-port equivalent T-network on port 2.

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        z11 = zmat[:,0,0]
        z12 = zmat[:,1,0]
        z22 = zmat[:,1,1]
        return Zfmodel(self.Fdata, z22-z12)
    
    def z_shunt_pi_1(self) -> Zfmodel:
        """Returns the shunt impedance for a 2-port equivalent 𝜋-network on port 1.

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        z11 = zmat[:,0,0]
        z12 = zmat[:,1,0]
        z22 = zmat[:,1,1]
        detz = z11*z22-z12*z12
        y11 = z22/detz
        y12 = -z12/detz
        y22 = z11/detz
        return Zfmodel(self.Fdata, 1/(y11+y12))

    def z_shunt_pi_2(self) -> Zfmodel:
        """Returns the series impedance for a 2-port equivalent 𝜋-network on port 2.

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        z11 = zmat[:,0,0]
        z12 = zmat[:,1,0]
        z22 = zmat[:,1,1]
        detz = z11*z22-z12*z12
        y11 = z22/detz
        y12 = -z12/detz
        y22 = z11/detz
        return Zfmodel(self.Fdata, 1/(y22+y12))
    
    def z_series_pi(self) -> Zfmodel:
        """Returns the series impedance for a 2-port equivalent 𝜋-network.

        Returns:
            Zfmodel: A class that can be called as LumpedElement function
        """
        zmat = self.generate_Z_param()
        z11 = zmat[:,0,0]
        z12 = zmat[:,1,0]
        z22 = zmat[:,1,1]
        detz = z11*z22-z12*z12
        y11 = z22/detz
        y12 = -z12/detz
        y22 = z11/detz
        return Zfmodel(self.Fdata, 1/(-y12))

    def z_series(self) -> Zfmodel:
        """Returns the single component series Z-parameter.

        If its a 1-port S-parameter file it will simply be Z12
        If its a 2-port S-parameter file it will assume its a singular matrix where
         - S12 = S21
         - S11 = S22
        In which case it will return the equivalent series impedance

        Returns:
            Zfmodel: _description_
        """
        if self.n_ports == 1:
            Z = self.generate_Z_param()
            return Zfmodel(self.Fdata, Z)
        elif self.n_ports == 2:
            logger.debug(' - Singular S-parameter file. Treating it as a single series impedance.')
            S = self.Sdata
            z0 = self.refimp

            s11 = S[:, 0, 0]
            s12 = S[:, 0, 1]
            s21 = S[:, 1, 0]
            s22 = S[:, 1, 1]
            
            B = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)
            
            return Zfmodel(self.Fdata, B)
    def generate_Z_param(self) -> np.ndarray:
        """Returns the (n_freq,n,n) Z-parameters of this S-parameter component

        Returns:
            np.ndarray: The Z-parameters
        """
        S = self.Sdata  # Shape (n_freq, n, n)
        Z0 = np.diag(self.refimp *np.ones(self.n_ports))
        
        n_freq, n, _ = S.shape
        
        # Create identity matrices for broadcasting: shape (1, n, n)
        I = np.eye(n)[np.newaxis, :, :]
        
        # Calculate Z = Z0 * (I + S) * inv(I - S)
        # Using np.linalg.inv for matrix inversion
        Z = Z0 @ (I + S) @ np.linalg.inv(I - S)
        
        return Z

    def S(self, i: int, j: int) -> np.ndarray:
        """Return the S-parameter array corresponding to the given port number

        Args:
            i (int): The output port
            j (int): The input Port

        Returns:
            np.ndarray: The S-parameter array
        """
        return self.Sdata[:, i-1, j-1].squeeze()
    
    def interp_S(self, f: np.ndarray) -> np.ndarray:
        """Interpolate the S-parameter matrix at the given frequencies.

        Args:
            f (np.ndarray): The frequencies in Hz. They must lie within the frequency range of the file.

        Returns:
            np.ndarray: The S-parameters with the shape (n_freq, n_ports, n_ports)
        """
        f = np.atleast_1d(f)
        fmin, fmax = np.min(self.Fdata), np.max(self.Fdata)
        tol = 1e-9 * fmax
        if np.min(f) < fmin - tol or np.max(f) > fmax + tol:
            raise ValueError(
                f"The Touchstone file covers {fmin/1e9:.6g} to {fmax/1e9:.6g} GHz, but S-parameters are needed "
                f"from {np.min(f)/1e9:.6g} to {np.max(f)/1e9:.6g} GHz. Use a file that covers the simulated frequencies."
            )
        f = np.clip(f, fmin, fmax)
        n = f.shape[0]
        nps = self.n_ports
        Sout = np.zeros((n,nps,nps), dtype=np.complex128)
        for i in range(nps):
            for j in range(nps):
                Sout[:,i,j] = interp1d(self.Fdata, self.Sdata[:,i,j])(f)
        return Sout
    

    def RemoveSP(self, touchstoneClass: 'TouchstoneData', port: int) -> 'TouchstoneData':
        """De-embed a 2-port fixture network from this touchstone measurement.

        Supports both 1-port and 2-port DUTs:
        - 1-port: solves Γ_dut from Γ_meas = S11 + S12*S21*Γ_dut / (1 - S22*Γ_dut)
        - 2-port: uses T-parameter de-embedding T_dut = inv(T_fix) @ T_meas

        Args:
            touchstoneClass (TouchstoneData): A 2-port touchstone object representing
                the fixture to de-embed (e.g. a female-female adapter).
            port (int): The port number (1-based) from which to de-embed.

        Returns:
            TouchstoneData: self, with S-parameters updated after de-embedding.
        """
        if touchstoneClass.n_ports != 2:
            raise ValueError('The fixture touchstone data must be a 2-port (s2p) file.')

        if port < 1 or port > self.n_ports:
            raise ValueError(f'Port {port} is out of range for a {self.n_ports}-port network.')

        # Interpolate fixture data onto our frequency axis
        fixture_S = np.zeros((self.n_frequencies, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                s_param = touchstoneClass.S(i + 1, j + 1)
                f_fixture = touchstoneClass.f
                re_interp = interp1d(f_fixture, s_param.real,
                                    kind=self.interp_type, fill_value='extrapolate')
                im_interp = interp1d(f_fixture, s_param.imag,
                                    kind=self.interp_type, fill_value='extrapolate')
                fixture_S[:, i, j] = re_interp(self.Fdata) + 1j * im_interp(self.Fdata)

        if self.n_ports == 1:
            # 1-port de-embedding using signal flow graph solution
            for iif in range(self.n_frequencies):
                S11f = fixture_S[iif, 0, 0]
                S12f = fixture_S[iif, 0, 1]
                S21f = fixture_S[iif, 1, 0]
                S22f = fixture_S[iif, 1, 1]

                gamma_meas = self.Sdata[iif, 0, 0]

                denom = S22f * (gamma_meas - S11f) + S12f * S21f
                if abs(denom) < 1e-30:
                    raise ValueError(f'De-embedding denominator near zero at freq index {iif}.')

                gamma_dut = (gamma_meas - S11f) / denom
                self.Sdata[iif, 0, 0] = gamma_dut

        elif self.n_ports == 2:
            for iif in range(self.n_frequencies):
                # Convert fixture S to T-parameters
                S = fixture_S[iif, :, :]
                if abs(S[1, 0]) < 1e-30:
                    raise ValueError(f'Fixture S21 near zero at freq index {iif}.')
                T_fix = np.array([
                    [-(S[0,0]*S[1,1] - S[0,1]*S[1,0]) / S[1,0], S[0,0] / S[1,0]],
                    [-S[1,1] / S[1,0], 1.0 / S[1,0]]
                ])

                # Convert measured S to T-parameters
                Sm = self.Sdata[iif, :, :]
                if abs(Sm[1, 0]) < 1e-30:
                    raise ValueError(f'Measured S21 near zero at freq index {iif}.')
                T_meas = np.array([
                    [-(Sm[0,0]*Sm[1,1] - Sm[0,1]*Sm[1,0]) / Sm[1,0], Sm[0,0] / Sm[1,0]],
                    [-Sm[1,1] / Sm[1,0], 1.0 / Sm[1,0]]
                ])

                # De-embed from the appropriate port
                if port == 1:
                    T_result = np.linalg.inv(T_fix) @ T_meas
                else:
                    T_result = T_meas @ np.linalg.inv(T_fix)

                # Convert T back to S
                if abs(T_result[0, 0]) < 1e-30:
                    raise ValueError(f'T11 near zero at freq index {iif}.')
                self.Sdata[iif, :, :] = np.array([
                    [T_result[0,1] / T_result[0,0],
                    (T_result[0,0]*T_result[1,1] - T_result[0,1]*T_result[1,0]) / T_result[0,0]],
                    [1.0 / T_result[0,0],
                    -T_result[1,0] / T_result[0,0]]
                ])
        else:
            raise ValueError(f'De-embedding is only supported for 1-port and 2-port DUTs, '
                            f'not {self.n_ports}-port.')

        return self