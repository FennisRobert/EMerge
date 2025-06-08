from emerge.solvers.fem.simmodel import _stitch_lists

lists = [[1,2,3],[5,4,3],[9,10,11],[5,6,7,8,9],[0,],[21,22,23,24]]

print(_stitch_lists(lists))