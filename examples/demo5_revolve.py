import emerge as em
"""This demo is still in progress. 

For now it just shows you how to work with the revolve system.
"""
model = em.Simulation('Revolve test')
model.check_version("0.6.7") # Checks version compatibility.

mm = 0.001
rad_feed = 30*mm
rad_out = 70*mm
len_horn = 100*mm
len_feed = 40*mm
th = 5*mm

poly = em.geo.XYPolygon([rad_feed, rad_feed, rad_out, 0, 0], [-len_feed, 0, len_horn, len_horn, -len_feed])
vol_in = poly.revolve(em.ZXPLANE.cs(), (0,0,0), (1,0,0))
poly = em.geo.XYPolygon([rad_feed+th, rad_feed+th, rad_out+th, 0, 0], [-len_feed, 0, len_horn, len_horn, -len_feed])
vol_out = poly.revolve(em.ZXPLANE.cs(), (0,0,0), (1,0,0))

ratio = 2.5
airbox = em.geo.pmlbox(40*mm, ratio*rad_out, ratio*rad_out, (len_horn-5*mm, -ratio*rad_out/2, -ratio*rad_out/2), thickness=30*mm, 
                       top=True, bottom=True, right=True, front=True, back=True)

model.view()
