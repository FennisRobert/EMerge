# import shapely as shp
# import numpy as np


# def rectangle(p1: tuple, p2: tuple) -> shp.Polygon:
#     x1, y1 = p1
#     x2, y2 = p2
#     x1, x2 = sorted([x1, x2])
#     y1, y2 = sorted([y1, y2])
#     return shp.Polygon(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))

# def rect_center(center: tuple, width: float, height: float):
#     x0, y0 = center
#     w, h = width, height
#     return shp.Polygon(((x0-w/2,y0-h/2),(x0+w/2,y0-h/2),(x0+w/2,y0+h/2),(x0-w/2,y0+h/2)))

# def polygon(points) -> shp.Polygon:
#     return shp.Polygon(points)

# def circle(center: tuple, radius: float, segments: int, ang_range: tuple = (0,360)):
#     x,y = center
#     points = []
#     path = []
#     angs = np.linspace(ang_range[0],ang_range[1],segments+1)

#     angs = angs*np.pi/180
#     for i in range(segments+1):
#         ang = angs[i]
#         points.append((x+radius*np.cos(ang),y+radius*np.sin(ang)))
#         path.append((x+radius*np.cos(ang),y+radius*np.sin(ang)))

#     if ang_range[0] != ang_range[1]%360:
#         points = [center,] + points
#     else:
#         points = points[:-1]
#     path = path + [path[0],]


#     return shp.Polygon(points), shp.LineString(path)

# def transform(polygon: shp.Polygon, transformation: callable, reverse=False) -> shp.Polygon:
#     ''' Transforms the provided shapely polygon by applying the transformation f(x,y) to each coordinate'''
#     ex, ey = polygon.exterior.xy
#     interiors = [interior.xy for interior in polygon.interiors]
#     new_poly = shp.Polygon(transformation(x,y) for x,y in zip(ex, ey))
#     for interior in interiors:
#         print(f'Type of interior = {type(interior)}, {interior}')
#         ix, iy = interior
#         new_poly = new_poly.difference(shp.Polygon(transformation(x,y) for x,y in zip(ix, iy)))
#     if reverse:
#         new_poly = new_poly.reverse()
#     return new_poly


# def move(polygon: shp.Polygon, dx: float, dy: float) -> shp.Polygon:
#     func = lambda x,y: (x+dx, y+dy)
#     return transform(polygon, func)

# def array(polygon: shp.Polygon, direction: tuple, N: int, include_original: bool = True) -> list[shp.Polygon]:
#     polys = []
#     for i in range(N+1):
#         polys.append(move(polygon, direction[0]*i, direction[1]*i))
#     if not include_original:
#         polys = polys[1:]
#     return polys

# def mirror(polygon: shp.Polygon, origin: tuple, axis: tuple) -> shp.Polygon:
#     ax, ay = axis[0]/np.sqrt(axis[0]**2 + axis[1]**2), axis[1]/np.sqrt(axis[0]**2 + axis[1]**2)
#     def _mir_transform(x, y):
#         dotprod = (x-origin[0])*ax + (y-origin[1])*ay
#         x2 = x - dotprod*ax*2
#         y2 = y - dotprod*ay*2
#         return x2, y2
#     return transform(polygon, _mir_transform, reverse=True)
    
# def rasterize(polygons: list[shp.Polygon], gridsize: float) -> list[shp.Polygon]:
#     output_polygons = []
#     grid = lambda x: np.round(np.array(x)/gridsize)*gridsize
#     for poly in polygons:
#         ex, ey = poly.exterior.xy
#         interiors = [interior.xy for interior in poly.interiors]
#         new_poly = shp.Polygon((x,y) for x,y in zip(grid(ex), grid(ey)))
#         for ix, iy in interiors:
#             new_poly.difference(shp.Polygon((x,y) for x,y in zip(grid(ix), grid(iy))))
#         output_polygons.append(new_poly)
#     return output_polygons