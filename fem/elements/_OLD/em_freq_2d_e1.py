import numpy as np
from ..bc import PEC, Port, BoundaryCondition, ABC
from ..plot import plot_mesh
from loguru import logger
c0 = 299792458

def select_bc(bcs: list[BoundaryCondition], bctype):
    return [bc for bc in bcs if isinstance(bc,bctype)]

def _iterate_segments(bcs: list[BoundaryCondition]) -> tuple[BoundaryCondition, list[np.ndarray]]:
    indices_list = []
    for bc in bcs:
        for indices in bc.node_indices:
            indices_list.append((bc,indices))
    return indices_list

@logger.catch
def assemble_base_matrix_Ez(vertices: np.ndarray, triangles: np.ndarray, er: np.ndarray, ur: np.ndarray, f0: float):
    nT = triangles.shape[1]
    nV = vertices.shape[1]

    k0 = 2 * np.pi * f0 / c0
    D = np.zeros((nV, nV)).astype(np.complex128)
    
    beta = -(k0**2)

    for it in range(nT):
        ax = er[it]
        ay = er[it]
        i, j, k = triangles[:, it]
        x1, y1 = vertices[:, i]
        x2, y2 = vertices[:, j]
        x3, y3 = vertices[:, k]

        M = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
        M2 = np.linalg.inv(M).T
        (m11, m12, m13), (m21, m22, m23), (m31, m32, m33) = M2
        A = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

        D[i, i] += 2 * A * ax * m11**2 / 2 + 2 * A * ay * m12**2 / 2 + 2 * A * beta / 12
        D[i, j] += (
            2 * A * ax * m11 * m21 / 2 + 2 * A * ay * m12 * m22 / 2 + 2 * A * beta / 24
        )
        D[i, k] += (
            2 * A * ax * m11 * m31 / 2 + 2 * A * ay * m12 * m32 / 2 + 2 * A * beta / 24
        )
        D[j, i] += (
            2 * A * ax * m11 * m21 / 2 + 2 * A * ay * m12 * m22 / 2 + 2 * A * beta / 24
        )
        D[j, j] += 2 * A * ax * m21**2 / 2 + 2 * A * ay * m22**2 / 2 + 2 * A * beta / 12
        D[j, k] += (
            2 * A * ax * m21 * m31 / 2 + 2 * A * ay * m22 * m32 / 2 + 2 * A * beta / 24
        )
        D[k, i] += (
            2 * A * ax * m11 * m31 / 2 + 2 * A * ay * m12 * m32 / 2 + 2 * A * beta / 24
        )
        D[k, j] += (
            2 * A * ax * m21 * m31 / 2 + 2 * A * ay * m22 * m32 / 2 + 2 * A * beta / 24
        )
        D[k, k] += 2 * A * ax * m31**2 / 2 + 2 * A * ay * m32**2 / 2 + 2 * A * beta / 12
    return D

@logger.catch
def assemble_eig_base_matrix_Ez(vertices: np.ndarray, triangles: np.ndarray, er: np.ndarray, ur: np.ndarray):
    nT = triangles.shape[1]
    nV = vertices.shape[1]

    D = np.zeros((nV, nV)).astype(np.complex128)
    B = np.zeros((nV, nV)).astype(np.complex128)
    

    for it in range(nT):
        ax = er[it]
        ay = er[it]
        i, j, k = triangles[:, it]
        x1, y1 = vertices[:, i]
        x2, y2 = vertices[:, j]
        x3, y3 = vertices[:, k]

        M = np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]])
        M2 = np.linalg.inv(M).T
        (m11, m12, m13), (m21, m22, m23), (m31, m32, m33) = M2
        A = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))

        D[i, i] += 2 * A * ax * m11**2 / 2 + 2 * A * ay * m12**2 / 2
        D[i, j] += (
            2 * A * ax * m11 * m21 / 2 + 2 * A * ay * m12 * m22 / 2
        )
        D[i, k] += (
            2 * A * ax * m11 * m31 / 2 + 2 * A * ay * m12 * m32 / 2
        )
        D[j, i] += (
            2 * A * ax * m11 * m21 / 2 + 2 * A * ay * m12 * m22 / 2
        )
        D[j, j] += 2 * A * ax * m21**2 / 2 + 2 * A * ay * m22**2 / 2
        D[j, k] += (
            2 * A * ax * m21 * m31 / 2 + 2 * A * ay * m22 * m32 / 2
        )
        D[k, i] += (
            2 * A * ax * m11 * m31 / 2 + 2 * A * ay * m12 * m32 / 2
        )
        D[k, j] += (
            2 * A * ax * m21 * m31 / 2 + 2 * A * ay * m22 * m32 / 2
        )
        D[k, k] += 2 * A * ax * m31**2 / 2 + 2 * A * ay * m32**2 / 2 
        

        B[i, i] += 2 * A / 12
        B[i, j] += 2 * A / 24
        B[i, k] += 2 * A / 24
        B[j, i] += 2 * A / 24
        B[j, j] += 2 * A / 12
        B[j, k] += 2 * A / 24
        B[k, i] += 2 * A / 24
        B[k, j] += 2 * A / 24
        B[k, k] += 2 * A / 12
    return D, B

@logger.catch
def assemble_matrix_Ez(vertices: np.ndarray, 
                     triangles: np.ndarray, 
                     boundary_normals: dict,
                     n_field_points: int,
                     er: np.ndarray, 
                     ur: np.ndarray, 
                     bcs: list[BoundaryCondition],
                     frequency: float):
    
    k0 = 2*np.pi*frequency/299792458
    logger.debug('Assembling base matrix')
    D = assemble_base_matrix_Ez(vertices, triangles, er, ur, frequency)

    logger.debug('Starting boundary conditions.')
    nv = vertices.shape[1]
    # plot_matrix_sparsity(D)
    D = D.astype(np.complex128)
    b = np.zeros((nv,)).astype(np.complex128)

    xs = vertices[0, :]
    ys = vertices[1, :]

    bcs = list(bcs.values())
    pecs = [bc for bc in bcs if isinstance(bc,PEC)]
    ports = [bc for bc in bcs if isinstance(bc,Port)]

    # Process all PEC Boundary Conditions
    pec_ids = []
    for pec in pecs:
        ids = pec.node_indices
        for i in ids:
            D[i, :] = 0
            D[:, i] = 0
        pec_ids += list(ids)

    # Process all port boundary Conditions
    for pids, port in _iterate_segments(ports):
        port.pids = pids

        x = xs[np.array(pids)]
        y = ys[np.array(pids)]
        #plot_mesh(vertices, triangles, highlight_vertices=pids)
        Ez, beta = port.port_mode(x, y, k0)

        port._field_amplitude = np.zeros_like(xs)
        port._field_amplitude[pids] = Ez

        if port.active:
            for i, j, ez1, ez2 in zip(pids[:-1], pids[1:], Ez[:-1], Ez[1:]):
                x1, x2 = xs[i], xs[j]
                y1, y2 = ys[i], ys[j]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                b[i] += -2 * 1j * beta * ez1 * length / 2
                b[j] += -2 * 1j * beta * ez2 * length / 2
        for i, j in zip(pids[:-1], pids[1:]):
            x1, x2 = xs[i], xs[j]
            y1, y2 = ys[i], ys[j]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            D[i, i] += -1j * beta * length / 3
            D[i, j] += -1j * beta * length / 6
            D[j, i] += -1j * beta * length / 6
            D[j, j] += -1j * beta * length / 3

    # pecids = [i for i in pecids if i not in portids]

    solve_ids = np.array([i for i in range(nv) if i not in pec_ids])
    
    return D, b, solve_ids

@logger.catch
def assemble_eig_matrix_Ez(vertices: np.ndarray, 
                     triangles: np.ndarray,
                     er: np.ndarray, 
                     ur: np.ndarray, 
                     bcs: list[BoundaryCondition]):
    
    logger.debug('Assembling base matrix')
    D, B = assemble_eig_base_matrix_Ez(vertices, triangles, er, ur)

    logger.debug('Starting boundary conditions.')
    nv = vertices.shape[1]
    # plot_matrix_sparsity(D)
    D = D.astype(np.complex128)

    pecs = [bc for bc in bcs if isinstance(bc,PEC)]
    
    # Process all PEC Boundary Conditions
    pec_ids = []
    for pec in pecs:
        id_set = pec.node_indices
        for ids in id_set:
            for i in ids:
                D[i, :] = 0
                D[:, i] = 0
                B[i, :] = 0
                B[:, i] = 0
            pec_ids += list(ids)

    # pecids = [i for i in pecids if i not in portids]

    solve_ids = np.array([i for i in range(nv) if i not in pec_ids])
    
    return D, B, solve_ids


def compute_sparam(pids, vertices, field, port_mode, active=False):
    Q = 0
    if active:
        Q = 1
    Ip = 0
    Sparam = 0
    for i1, i2 in zip(pids[:-1], pids[1:]):
        x1, y1 = vertices[:, i1]
        x2, y2 = vertices[:, i2]
        ezp1 = port_mode[i1]
        ezp2 = port_mode[i2]
        ez1 = field[i1]
        ez2 = field[i2]
        L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        Sparam += L * ((ez1 - Q*ezp1) * ezp1.conj() + (ez2 - Q*ezp2) * ezp2.conj()) / 2
        Ip += L * (ezp1 * ezp1.conj() + ezp2 * ezp2.conj()) / 2
    Sparam = Sparam/Ip
    return Sparam
