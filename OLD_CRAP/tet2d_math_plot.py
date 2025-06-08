import numpy as np
import pyescher as pe


xs = np.array([0, 1, 0.5])
ys = np.array([0, 0, 1])

x1, x2, x3 = xs
y1, y2, y3 = ys


xp = np.linspace(0,1,21)
yp = np.linspace(0,1,21)
xp, yp = np.meshgrid(xp, yp)
xp = xp.flatten()
yp = yp.flatten()

e1x = x2 - x1
e1y = y2 - y1
e2x = x3 - x1
e2y = y3 - y1

basis = np.linalg.pinv(np.array([[e1x, e2x], [e1y, e2y]]))

coords = np.array([xp, yp])
coords_local = basis @ (coords - np.array([[x1], [y1]]))
inside_triangle = (coords_local[0,:] + coords_local[1,:] <= 1) & (coords_local[0,:] >= 0) & (coords_local[1,:] >= 0)

x = xp[inside_triangle]
y = yp[inside_triangle]

A= 0.5*(x2*y3 - x3*y2 + x1*(y2 - y3) + x3*(y1 - y2) + x2*(y3 - y1))
Ne1x = -((y1 - y2)*(x*(y2 - y3) + x2*y3 - x3*y2 - y*(x2 - x3)) - (y2 - y3)*(x*(y1 - y2) + x1*y2 - x2*y1 - y*(x1 - x2)))*(x*(y1 - y3) + x1*y3 - x3*y1 - y*(x1 - x3))/(8*A**3*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)**3)
Ne1y = ((x1 - x2)*(x*(y2 - y3) + x2*y3 - x3*y2 - y*(x2 - x3)) - (x2 - x3)*(x*(y1 - y2) + x1*y2 - x2*y1 - y*(x1 - x2)))*(x*(y1 - y3) + x1*y3 - x3*y1 - y*(x1 - x3))/(8*A**3*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)**3)
Ne2x = ((y1 - y3)*(x*(y2 - y3) + x2*y3 - x3*y2 - y*(x2 - x3)) - (y2 - y3)*(x*(y1 - y3) + x1*y3 - x3*y1 - y*(x1 - x3)))*(x*(y1 - y2) + x1*y2 - x2*y1 - y*(x1 - x2))/(8*A**3*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)**3)
Ne2y = -((x1 - x3)*(x*(y2 - y3) + x2*y3 - x3*y2 - y*(x2 - x3)) - (x2 - x3)*(x*(y1 - y3) + x1*y3 - x3*y1 - y*(x1 - x3)))*(x*(y1 - y2) + x1*y2 - x2*y1 - y*(x1 - x2))/(8*A**3*(x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)**3)

#pe.plot2D(pe.ScatterPlot(xs, ys), pe.VectorPlot(x, y, Ne1x, Ne1y, color='blue'), pe.VectorPlot(x,y,Ne2x, Ne2y, color='red'))
a = 0
b = 1
pe.plot2D(pe.ScatterPlot(xs, ys), pe.VectorPlot(x, y, a*Ne1x+b*Ne2x, a*Ne1y+b*Ne2y, color='blue'))