# from ..domain import Geometry, Edge, Point
# from ..material import Material
# from ..mesh import Mesh2D
# from ..old_dataset import Dataset2D
# from ..bc import BoundaryCondition, PEC, RectangularWaveguide
# import numpy as np
# from typing import Callable
# from enum import Enum
# from scipy import sparse
# from loguru import logger


# class EMFreqDomain2D:
#     def __init__(self, geo: Geometry):
#         self.frequencies: list[float] = []
#         self.current_frequency = 0
#         self.geo: Geometry = geo
#         self.boundary_conditions: list[BoundaryCondition] = []
#         self.solution: Dataset2D = None

#     @property
#     def nports(self) -> int:
#         return len([bc for bc in self.boundary_conditions if isinstance(bc,RectangularWaveguide)])
    
#     def ports(self) -> list[BoundaryCondition]:
#         return sorted([bc for bc in self.boundary_conditions if isinstance(bc,RectangularWaveguide)], key=lambda x: x.port_number)
    
#     @logger.catch
#     def _initialize_bcs(self) -> None:
#         logger.info('Initializing boundary conditions.')
#         self.boundary_conditions = []
#         tags = []
#         for tag, edge in self.geo.edges.tag_to_edge.items():
#             if not self.geo.is_boundary(edge):
#                 continue
#             tags.append(tag)
#         pec = PEC([(1, tag) for tag in tags])
#         self.boundary_conditions.append(pec)

#     @logger.catch
#     def assign(self, bc: BoundaryCondition, edge: Edge = None, edges: list[Edge] = None, point: Point = None, points: list[Point] = None) -> BoundaryCondition:
#         if edge is not None:
#             logger.info(f'Assinging the boundary condition {bc} to edge {edge}')
#             bc.add_tags([edge.dimtag,])
#         if edges is not None:
#             logger.info(f'Assigning the boundary condition {bc} to edges {edges}')
#             bc.add_tags([e.dimtag for e in edges])
        
#         if point is not None:
#             logger.info(f'Assinging the boundary condition {bc} to point {point}')
#             bc.add_nodes([point.dimtag,])
#         if points is not None:
#             logger.info(f'Assinging the boundary condition {bc} to points {points}')
#             bc.add_nodes([p.dimtag for p in points])

#         logger.info('Clearing other boundary conditions')
#         for existing_bc in self.boundary_conditions:
#             excluded = existing_bc.exclude_bc(bc)
#             if excluded:
#                 logger.warning(f'Overwritten the following edges: {excluded} from {existing_bc}')
#         self.boundary_conditions.append(bc)
#         return bc

#     def set_frequency(self, frequency) -> None:
#         logger.info(f'Setting frequency as {frequency/1e6}MHz.')
#         if isinstance(frequency, (tuple, list, np.ndarray)):
#             self.frequencies = list(frequency)
#         else:
#             self.frequencies = [frequency]

#     def iter_frequencies(self):
#         for i, frequency in enumerate(self.frequencies):
#             self.current_frequency = i
#             yield i, frequency

#     def get_discretizer(self) -> Callable:
#         def disc(material: Material):
#             return 299792456/(max(self.frequencies)* np.abs(material.neff))
#         return disc
    
#     @logger.catch
#     def solve(self, mesh: Mesh2D, solver = None):
#         if solver is None:
#             logger.info('Defaulting to BiCGStab.')
#             solver = sparse.linalg.bicgstab

#         if mesh.element_order==1:
#             logger.info('Detected 1st order elements.')
#             from ..elements.em_freq_2d_e1 import assemble_matrix_Ez, compute_sparam
#             vertices = mesh.vertices
#             triangles = mesh.triangles
#             boundary_normals = mesh.boundary_normals

#         elif mesh.element_order==2:
#             logger.info('Detected 2nd order elements.')
#             from ..elements.em_freq_2d_e2 import assemble_matrix_Ez, compute_sparam
#             vertices = mesh.quad_vertices
#             triangles = mesh.quad_elements
#             boundary_normals = mesh.quad_boundary_normals
        
#         er = self.geo.retreive(lambda mat,x,y: mat.fer(x,y))
#         ur = self.geo.retreive(lambda mat,x,y: mat.fur(x,y))
        
#         dataset = Dataset2D(self.frequencies, mesh.mvertices, mesh.mtriangles, nports=self.nports)
#         dataset.er = er
#         dataset.ur = ur
#         logger.info('Iterating frequencies.')
#         for i, freq in self.iter_frequencies():
#             logger.info(f'Frequency = {freq/1e9:.3f} GHz')
#             k0 = 2*np.pi*freq/299792458
#             M, b, solve_ids, aux_data = assemble_matrix_Ez(vertices, triangles, boundary_normals, mesh.nF, er, ur, self.boundary_conditions, freq)

#             logger.info('Selecting solvable ids')
#             Msol = sparse.csc_matrix(M[np.ix_(solve_ids, solve_ids)])
#             bsol = b[solve_ids]
            
#             solution = np.zeros((mesh.nF,), dtype=np.complex128)

#             logger.info(f'Freq={freq*1e-6:.0f}MHz - using {solver}')
#             solution[solve_ids], info = solver(Msol, bsol)

#             #print(solution)
#             dataset.Ez[i,:] = solution
            
#             dataset.Hx[i,:], dataset.Hy[i,:] = self._compute_hxy(dataset.Ez[i,:], vertices, triangles, freq)

#             logger.info(f'Computing S-Parameters')
#             for ip, port in enumerate(self.ports()):
#                 S = compute_sparam(port.all_nodes, vertices, dataset.Ez[i,:], dataset.Hx[i,:], dataset.Hy[i,:], port._field_amplitude, port.active)
#                 dataset.S[i,ip,0] = S
#             if len(self.ports()) > 0:
#                 dataset.S[i,:,0] = dataset.S[i,:,0]#/np.sqrt(np.sum(np.abs(dataset.S[i,:,0])**2))

#             logger.info('Computation complete!')
#         # Make Reciprocal

#         self.solution = dataset
    
#     @logger.catch
#     def eigenmode(self, mesh: Mesh2D, solver = None, num_sols: int = 6):
        
#         if solver is None:
#             logger.info('Defaulting to BiCGStab.')
#             solver = sparse.linalg.eigs

#         if mesh.element_order in (1,2):
#             logger.info('Detected 1st order elements.')
#             from ..elements.em_freq_2d_e1 import assemble_eig_matrix_Ez
#             vertices = mesh.vertices
#             triangles = mesh.triangles
#             boundary_normals = mesh.boundary_normals

#         # elif mesh.element_order==2:
#         #     logger.info('Detected 2nd order elements.')
#         #     from ..physics_compilers.em_freq_2d_e2 import assemble_matrix_Ez, compute_sparam
#         #     vertices = mesh.quad_vertices
#         #     triangles = mesh.quad_elements
#         #     boundary_normals = mesh.quad_boundary_normals
        
#         er = self.geo.retreive(lambda mat,x,y: mat.fer2d(x,y))
#         ur = self.geo.retreive(lambda mat,x,y: mat.fur2d(x,y))
        
#         dataset = Dataset2D(self.frequencies, mesh.mvertices, mesh.mtriangles, nports=self.nports)
#         dataset.er = er
#         dataset.ur = ur
#         logger.info('Iterating frequencies.')
        

#         A, B, solvenodes = assemble_eig_matrix_Ez(mesh.vertices, mesh.triangles, er, ur, self.boundary_conditions)

#         A = A[np.ix_(solvenodes, solvenodes)]
#         B = B[np.ix_(solvenodes, solvenodes)]
#         A = sparse.csc_matrix(A)
#         B = sparse.csc_matrix(B)
        
#         w, v = sparse.linalg.eigs(A, k=num_sols, M=B, which='SM')
#         logger.info(f'Eigenvalues: {w}')
#         Esol = np.zeros((num_sols, mesh.vertices.shape[1]), dtype=np.complex128)
#         Esol[:, solvenodes] = v.T
#         #Esol[:, :] = v.T
#         dataset.Ez = Esol

#         self.solution = dataset
    
#     @staticmethod
#     @logger.catch
#     def _compute_hxy(Ez, vertices, triangles,freq):
#         counter = np.zeros_like(Ez)
#         Hsx = np.zeros_like(Ez).astype(np.complex128)
#         Hsy = np.zeros_like(Ez).astype(np.complex128)
#         for it in range(triangles.shape[1]):
#             ids = triangles[:,it]
#             xs = vertices[0,ids]
#             x1, x2, x3, x4, x5, x6 = xs
#             ys = vertices[1,ids]
#             y1, y2, y3, y4, y5, y6 = ys
#             M = np.array([[x**2, y**2, x*y, x, y, 1] for x, y in zip(xs, ys)])
#             iM = np.linalg.pinv(M)

#             a, b, c, d, e, f = iM @ Ez[ids]
#             #Ez = a*x**2 + b*y**2 + c*x*y + d*x + e*y + f
#             #Hx = dEz/dy
#             #Hy = -dEz/dx

#             Hx = +(2*b*ys + c*xs + e)
#             Hy = -(2*a*xs + c*ys + d)
#             Hsx[ids] += Hx
#             Hsy[ids] += Hy
#             counter[ids] += 1

#         w0 = 2*np.pi*freq
#         counter[counter==0] = 1
#         u0 = 4*np.pi*1e-7
#         Hsx = -Hsx/(counter*1j*w0*u0)
#         Hsy = -Hsy/(counter*1j*w0*u0)
#         return Hsx, Hsy

