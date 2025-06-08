import numpy as np
from ..bc import PEC, Port, BoundaryCondition, ABC, PointSource
from ..plot import plot_mesh

from loguru import logger

c0 = 299792458

def select_bc(bcs: list[BoundaryCondition], bctype):
    output = [bc for bc in bcs if isinstance(bc,bctype)]
    return output

def dirdiff(func: callable, x: float, y: float, direction: tuple[float,float], order: int = 1, eps: float = 1e-9):
    dx, dy = direction[0]*eps, direction[1]*eps
    if order==1:
        return (func(x+dx,y+dy) - func(x,y))/eps
    elif order==2:
        return (func(x+dx,y+dy) - 2 * func(x,y) + func(x-dx,y-dy))/(eps**2)
    
@logger.catch
def _iterate_segments(bcs: list[BoundaryCondition]) -> list[tuple[BoundaryCondition, np.ndarray]]:
    indices_list = []
    for bc in bcs:
        for indices in bc.indices:
            indices_list.append((bc,indices))
    return indices_list

@logger.catch
def assemble_base_matrix_Ez(vertices: np.ndarray, triangles: np.ndarray, N: int, er: np.ndarray, ur: np.ndarray, f0: float):
    ''' This function assembles the base matrix for the solution of the wave equation in 2D'''
    nT = triangles.shape[1]

    k0 = 2 * np.pi * f0 / c0
    D = np.zeros((N, N)).astype(np.complex128)
    
    beta = -(k0**2)

    for it in range(nT):
        ax = 1/er[it]
        ay = 1/er[it]
        i1, j1, k1, i2, j2, k2 = triangles[:, it]
        x1, y1 = vertices[:, i1]
        x2, y2 = vertices[:, j1]
        x3, y3 = vertices[:, k1]

        x1sq = x1**2
        x2sq = x2**2
        x3sq = x3**2
        y1sq = y1**2
        y2sq = y2**2
        y3sq = y3**2
        x1x2 = x1*x2
        x1x3 = x1*x3
        x2x3 = x2*x3
        y1y2 = y1*y2
        y1y3 = y1*y3
        y2y3 = y2*y3
        A = 0.5 * np.abs((x1 - x3) * (y2 - y1) - (x1 - x2) * (y3 - y1))
        A2 = 2*A
        T0 = (y2sq - 2*y2y3 + y3sq)
        T1 = A2*ax
        T2 = A2*2*ax
        T3 = A2*4*ax
        T4 = (y1y2 - y1y3 - y2sq + y2y3)
        T5 = (-y1y2 + y1y3 + y2sq - y2y3)
        T6 = (y1y2 - y1y3 - y2y3 + y3sq)
        T7 = (-y1y2 + y1y3 + y2y3 - y3sq)
        T8 = (-y1sq + y1y2 + y1y3 - y2y3)
        T9 = (y1sq - y1y2 - y1y3 + y2sq - y2y3 + y3sq)
        TA = (x1sq*y2sq - 2*x1sq*y2y3 + x1sq*y3sq - 2*x1x2*y1y2 + 2*x1x2*y1y3 + 2*x1x2*y2y3 - 2*x1x2*y3sq + 2*x1x3*y1y2 - 2*x1x3*y1y3 - 2*x1x3*y2sq + 2*x1x3*y2y3 + x2sq*y1sq - 2*x2sq*y1y3 + x2sq*y3sq - 2*x2x3*y1sq + 2*x2x3*y1y2 + 2*x2x3*y1y3 - 2*x2x3*y2y3 + x3sq*y1sq - 2*x3sq*y1y2 + x3sq*y2sq)
        TB = (-x1sq + x1x2 + x1x3 - x2x3)
        TC = (-x1x2 + x1x3 + x2x3 - x3sq)
        TD = (x1x2 - x1x3 - x2sq + x2x3)
        TE = (x1sq - x1x2 - x1x3 + x2sq - x2x3 + x3sq)
        TF = (x1x2 - x1x3 - x2x3 + x3sq)
        TH = (-x1x2 + x1x3 + x2sq - x2x3)
        TI = (x1sq - x1x2 - x1x3 + x2x3)
        mA2beta = A2*-beta
        A22ay = A2*2*ay
        A24ay = A2*4*ay
        TA2 = 2*TA
        TA3 = 3*TA
        TA6 = 6*TA
        M1 = T1*T6/TA6 + A2*ay*TF/TA6 + mA2beta/360
        M2 = T1*T5/TA6 + A2*ay*TH/TA6 + mA2beta/360
        M3 = T2*T7/TA3 + A22ay*TC/TA3
        M4 = T2*T4/TA3 + A22ay*TD/TA3
        M5 = T1*(y1sq - y1y2 - y1y3 + y2y3)/TA6 + A2*ay*TI/TA6 + mA2beta/360
        M6 = T3*T9/TA3 + A24ay*TE/TA3 + A2*4*beta/45
        M7 = T3*T4/TA3 + A24ay*TD/TA3 + A2*2*beta/45
        M8 = T3*T8/TA3 + A24ay*TB/TA3 + A2*2*beta/45
        M9 = T2*T8/TA3 + A22ay*TB/TA3
        M10 = T3*T7/TA3 + A24ay*TC/TA3 + A2*2*beta/45
        M0 = mA2beta/90
        D[i1,i1] += T1*T0/TA2 + A2*ay*(x2sq - 2*x2x3 + x3sq)/TA2 + A2*beta/60
        D[i1,j1] += M1
        D[i1,k1] += M2
        D[i1,i2] += M3
        D[i1,j2] += M0
        D[i1,k2] += M4
        D[j1,i1] += M1
        D[j1,j1] += T1*(y1sq - 2*y1y3 + y3sq)/TA2 + A2*ay*(x1sq - 2*x1x3 + x3sq)/TA2 + A2*beta/60
        D[j1,k1] += M5
        D[j1,i2] += M3
        D[j1,j2] += M9
        D[j1,k2] += M0
        D[k1,i1] += M2
        D[k1,j1] += M5
        D[k1,k1] += T1*(y1sq - 2*y1y2 + y2sq)/TA2 + A2*ay*(x1sq - 2*x1x2 + x2sq)/TA2 + A2*beta/60
        D[k1,i2] += M0
        D[k1,j2] += M9
        D[k1,k2] += M4
        D[i2,i1] += M3
        D[i2,j1] += M3
        D[i2,k1] += M0
        D[i2,i2] += M6
        D[i2,j2] += M7
        D[i2,k2] += M8
        D[j2,i1] += M0
        D[j2,j1] += M9
        D[j2,k1] += M9
        D[j2,i2] += M7
        D[j2,j2] += M6
        D[j2,k2] += M10
        D[k2,i1] += M4
        D[k2,j1] += M0
        D[k2,k1] += M4
        D[k2,i2] += M8
        D[k2,j2] += M10
        D[k2,k2] += M6
    return D


@logger.catch
def assemble_matrix_Ez(vertices: np.ndarray, 
                     element_ids: np.ndarray, 
                     qbns: dict,
                     N: int,
                     er: np.ndarray, 
                     ur: np.ndarray, 
                     bcs: list[BoundaryCondition],
                     frequency: float):
    
    auxdata = dict()

    k0 = 2*np.pi*frequency/299792458
    logger.debug('Assembling base matrix')
    D = assemble_base_matrix_Ez(vertices, element_ids, N, er, ur, frequency)

    logger.debug('Implementing boundary conditions')
    nv = vertices.shape[1]

    D = D.astype(np.complex128)
    b = np.zeros((nv,)).astype(np.complex128)

    xs = vertices[0, :]
    ys = vertices[1, :]

    pecs = select_bc(bcs, PEC)
    ports = select_bc(bcs, Port)
    absorbing_boundary_conditions = select_bc(bcs, ABC)

    # Process all PEC Boundary Conditions
    pec_ids = []
    for pec,ids in _iterate_segments(pecs):
        logger.debug(f'PEC indices: {ids}')
        for i1 in ids:
            D[i1, :] = 0
            D[:, i1] = 0
        pec_ids += list(ids)

    # Process all port boundary Conditions
    for port, pids in _iterate_segments(ports):
        logger.debug(f'PORT indices: {pids}')
        port.pids = pids

        x = xs[np.array(pids)]
        y = ys[np.array(pids)]

        #plot_mesh(vertices, triangles, highlight_vertices=pids)

        Ez, beta = port.port_mode(x, y, k0)

        port._field_amplitude = np.zeros_like(xs)
        port._field_amplitude[pids] = Ez

        if port.active:
            for i1, i2, i3, ez1, ez2, ez3 in zip(pids[:-2:2], pids[1:-1:2], pids[2::2], Ez[:-2:2], Ez[1:-1:2], Ez[2::2]):
                x1, x3 = xs[i1], xs[i3]
                y1, y3 = ys[i1], ys[i3]
                length = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
                b[i1] += 2 * 1j * beta * ez1 * length / 6
                b[i2] += 2 * 1j * beta * ez2 * length * 2 / 3
                b[i3] += 2 * 1j * beta * ez3 * length / 6

        for i1, i2, i3 in zip(pids[:-2:2], pids[1:-1:2], pids[2::2]):
            x1, x2 = xs[i1], xs[i3]
            y1, y2 = ys[i1], ys[i3]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            D[i1,i1] += 1j * beta * length * 2/15
            D[i1,i2] += 1j * beta * length * 1/15
            D[i1,i3] += 1j * beta * length * -1/30
            D[i2,i1] += 1j * beta * length * 1/15
            D[i2,i2] += 1j * beta * length * 8/15
            D[i2,i3] += 1j * beta * length * 1/15
            D[i3,i1] += 1j * beta * length * -1/30
            D[i3,i2] += 1j * beta * length * 1/15
            D[i3,i3] += 1j * beta * length * 2/15

    ## TODO MAKE DEPENDENT ON MATERIAL PROPERTIES AS ALPHA = ER

    alpha = 1
    for abc,ids in _iterate_segments(absorbing_boundary_conditions):
        logger.debug(f'ABC indices: {ids}')
        x = xs[np.array(ids)]
        y = ys[np.array(ids)]

        for i1, i2, i3 in zip(ids[:-2:2], ids[1:-1:2], ids[2::2]):
            x1, x2, x3 = xs[i1], xs[i2], xs[i3]
            y1, y2, y3 = ys[i1], ys[i2], ys[i3]
            length = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
            
            dsx = x3-x1
            dsy = y3-y1
            l = np.sqrt(dsx**2+dsy**2)
            dsx = dsx/l
            dsy = dsy/l
            d1, d2, d3 = 0, l/2, l
            nx, ny = qbns[(i1, i2)]
            dsx, dsy = -ny, nx
            K = 1/np.sqrt(x2**2+y2**2)

            fnc = lambda x,y: abc.func(x,y,nx,ny,k0)
            psi1 = fnc(x1,y1)
            psi2 = fnc(x2,y2)
            psi3 = fnc(x3,y3)

            #M = np.array([[d1**2, d1, 1],[d2**2, d2, 1],[d3**2, d3, 1]])
            #abcm = np.linalg.pinv(M) @ np.array([psi1, psi2, psi3])
            
            # First derivative along the normal
            dpsin1 = dirdiff(fnc, x1, y1, (nx,ny), 1)
            dpsin2 = dirdiff(fnc, x2, y2, (nx,ny), 1)
            dpsin3 = dirdiff(fnc, x3, y3, (nx,ny), 1)

            #Second derivative along the line
            ddpsis1 = dirdiff(fnc, x1, y1, (dsx, dsy), 2)
            ddpsis2 = dirdiff(fnc, x2, y2, (dsx, dsy), 2)
            ddpsis3 = dirdiff(fnc, x3, y3, (dsx, dsy), 2) 

            if abc.order == 2:
                q1 = alpha*dpsin1 + alpha*(1j*k0+K/2-(1j*K**2)/(8*(1j*K-k0)))*psi1 - 1j*alpha/(2*(1j*K-k0))*ddpsis1
                q2 = alpha*dpsin2 + alpha*(1j*k0+K/2-(1j*K**2)/(8*(1j*K-k0)))*psi2 - 1j*alpha/(2*(1j*K-k0))*ddpsis2
                q3 = alpha*dpsin3 + alpha*(1j*k0+K/2-(1j*K**2)/(8*(1j*K-k0)))*psi3 - 1j*alpha/(2*(1j*K-k0))*ddpsis3

            else:
                q1 = alpha*dpsin1 + alpha*(1j*k0)*psi1
                q2 = alpha*dpsin2 + alpha*(1j*k0)*psi2
                q3 = alpha*dpsin3 + alpha*(1j*k0)*psi3
            
            # Q = 0
            # b[i1] += Q * q1 * length / 6
            # b[i2] += Q * q2 * length * 4 / 6
            # b[i3] += Q * q3 * length / 6

        for i,k,j in zip(ids[:-2:2], ids[1:-1:2], ids[2::2]):
            x0, y0 = abc.origin
            x1, x2 = xs[i], xs[j]
            y1, y2 = ys[i], ys[j]
            l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            K = 1/np.sqrt((x2-x0)**2+(y2-y0)**2)
            gamma = alpha * 1j * k0 + K/2
            gamma1 = alpha * (1j*k0 + K/2)# + (1j*K**2)/(8*(1j*K-k0)))
            gamma2 = -1j*alpha/(2*1j*k0+K)#-1j*alpha/(2*1j*K-k0))
            if abc.order==1:
                D[i,i] += 2*gamma*l/15
                D[i,k] += gamma*l/15
                D[i,j] += -gamma*l/30
                D[k,i] += gamma*l/15
                D[k,k] += 8*gamma*l/15
                D[k,j] += gamma*l/15
                D[j,i] += -gamma*l/30
                D[j,k] += gamma*l/15
                D[j,j] += 2*gamma*l/15
            else:
                D[i,i] += (2*gamma1*l**2 - 35*gamma2)/(15*l)
                D[i,k] += (gamma1*l**2 + 40*gamma2)/(15*l)
                D[i,j] += -gamma1*l/30 - gamma2/(3*l)
                D[k,i] += (gamma1*l**2 + 40*gamma2)/(15*l)
                D[k,k] += 8*(gamma1*l**2 - 10*gamma2)/(15*l)
                D[k,j] += (gamma1*l**2 + 40*gamma2)/(15*l)
                D[j,i] += -gamma1*l/30 - gamma2/(3*l)
                D[j,k] += (gamma1*l**2 + 40*gamma2)/(15*l)
                D[j,j] += (2*gamma1*l**2 - 35*gamma2)/(15*l)


    for bc in select_bc(bcs, PointSource):
        for index in bc.node_indices:
            b[index] += 1j*k0
            D[index,index] += 1j * k0

    auxdata['pecids'] = pec_ids
    solve_ids = np.array([i for i in range(nv) if i not in pec_ids])
    
    return D, b, solve_ids, auxdata

@logger.catch
def compute_sparam(pids, vertices, Ez, Hx, Hy, port_mode, active=False):
    from emerge.plot import Line, eplot
    Q = 0
    if active:
        Q = 1
    Ip = 0
    Sparam = 0
    npids = np.array(pids)
    xs = vertices[0,np.array(pids)]
    ys = vertices[1,np.array(pids)]
    x0, y0 = xs[0], ys[0]
    ds = np.sqrt((xs-x0)**2+(ys-y0)**2)

    #eplot([Line(ds, Ez[npids].real, name='Actual field'),Line(ds, Ez[npids].imag, name='Actual field'), Line(ds, port_mode[npids].real, name='Port Field')])
    
    total_power = 0

    for i1, i2, i3 in zip(pids[:-2:2], pids[1:-1:2], pids[2::2]):
        x1, y1 = vertices[:, i1]
        #x2, y2 = vertices[:, i2]
        x3, y3 = vertices[:, i3]

        dsx, dsy = (x3-x1), (y3-y1)
        ezp1 = port_mode[i1]
        ezp2 = port_mode[i2]
        ezp3 = port_mode[i3]
        ez1 = Ez[i1]
        ez2 = Ez[i2]
        ez3 = Ez[i3]

        L = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        # nx, ny = dsy/L, -dsx/L
        # Sx1 = 0.5*(ez1*np.conj(Hx[i1]))
        # Sy1 = 0.5*(-ez1*np.conj(Hy[i1]))
        # Sx2 = 0.5*(ez2*np.conj(Hx[i2]))
        # Sy2 = 0.5*(-ez2*np.conj(Hy[i2]))
        # Sx3 = 0.5*(ez3*np.conj(Hx[i3]))
        # Sy3 = 0.5*(-ez3*np.conj(Hy[i3]))
        #total_power += np.abs((Sx1*nx + Sy1*ny)/6 + (Sx2*nx + Sy2*ny)*2/3 + (Sx3*nx+Sy3*ny)/6)
        #total_power = 1
        #print('Total power:',total_power)
        Sparam += L * ((ez1 - Q*ezp1) * ezp1.conj()/6 
                       + (ez2 - Q*ezp2) * ezp2.conj()*2/3 
                       + (ez3 - Q*ezp3) * ezp3.conj()/6)
        Ip += L * (ezp1 * ezp1.conj()/6 
                   + ezp2 * ezp2.conj()*2/3 
                   + ezp3 * ezp3.conj()/6)
    Sparam = Sparam/Ip#/np.sqrt(total_power)
    return Sparam