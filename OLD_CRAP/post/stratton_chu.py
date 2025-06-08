import numpy as np
from ...OLD_CRAP.mesh import Mesh2D
from ..simmodel import Simulation2D
from emerge.plot import Line,eplot
from scipy.special import hankel2

def compute_farfield(angles: np.ndarray, 
                     Ez: np.ndarray, 
                     Hx: np.ndarray, 
                     Hy: np.ndarray, 
                     mesh: Mesh2D, 
                     boundary_tags: list[int], 
                     k0: float):

    farfield = np.zeros_like(angles).astype(np.complex128)

    lam = 2*np.pi/k0
    Q = np.sqrt(lam)*1j*k0/(4*np.pi)

    rx = np.cos(angles)
    ry = np.sin(angles)
    kx = k0*rx
    ky = k0*ry

    Z0 = 376.73031341259

    ANGLES = []
    EZV = []

    for tag in boundary_tags:
        ids = mesh.get_edge(tag)
        #mesh.plot_mesh(highlight_vertices=ids)
        for i1, i2, i3 in zip(ids[:-2:2], ids[1:-1:2], ids[2::2]):
            Ez1, Ez2, Ez3 = Ez[i1], Ez[i2], Ez[i3]
            Hx1, Hx2, Hx3 = Hx[i1], Hx[i2], Hx[i3]
            Hy1, Hy2, Hy3 = Hy[i1], Hy[i2], Hy[i3]

            x1, x2, x3 = mesh.quad_vertices[0,np.array([i1, i2, i3])]
            y1, y2, y3 = mesh.quad_vertices[1,np.array([i1, i2, i3])]

            ANGLES.append(np.atan2(y1,x1))
            ANGLES.append(np.atan2(y2,x2))
            EZV.append(Ez1)
            EZV.append(Ez2)
            nx, ny = mesh.quad_boundary_normals[(i1, i2)]

            l = np.sqrt((x3-x1)**2 + (y3-y1)**2)
            #AxBx = AyBz - AzBy
            #AxBy = AzBx - AxBz
            #AxBz = AxBy - AyBx

            nxHz1 = nx*Hy1 - ny*Hx1
            nxHz2 = nx*Hy2 - ny*Hx2
            nxHz3 = nx*Hy3 - ny*Hx3

            nxEx1 = ny*Ez1
            nxEy1 = -nx*Ez1

            nxEx2 = ny*Ez2
            nxEy2 = -nx*Ez2

            nxEx3 = ny*Ez3
            nxEy3 = -nx*Ez3

            rxnxHx1 = ry*nxHz1
            rxnxHy1 = -rx*nxHz1

            rxnxHx2 = ry*nxHz2
            rxnxHy2 = -rx*nxHz2

            rxnxHx3 = ry*nxHz3
            rxnxHy3 = -rx*nxHz3

            Intx1 = (nxEx1 - Z0*rxnxHx1) * np.exp(1j*(kx*x1 + ky*y1))
            Intx2 = (nxEx2 - Z0*rxnxHx2) * np.exp(1j*(kx*x2 + ky*y2))
            Intx3 = (nxEx3 - Z0*rxnxHx3) * np.exp(1j*(kx*x3 + ky*y3))

            Inty1 = (nxEy1 - Z0*rxnxHy1) * np.exp(1j*(kx*x1 + ky*y1))
            Inty2 = (nxEy2 - Z0*rxnxHy2) * np.exp(1j*(kx*x2 + ky*y2))
            Inty3 = (nxEy3 - Z0*rxnxHy3) * np.exp(1j*(kx*x3 + ky*y3))
            
            Intx = Intx1/6 + Intx2*2/3 + Intx3/6
            Inty = Inty1/6 + Inty2*2/3 + Inty3/6

            farfield += l * Q * (rx * Inty - ry * Intx)

    angs = np.array(ANGLES)
    ez = np.array(EZV)
    lin1 = Line(angs, np.abs(ez))
    #lin2 = Line(angs, np.imag(ez))
    eplot([lin1,])
    return farfield



