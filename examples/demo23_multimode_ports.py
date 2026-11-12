""" MULTI-MODE WAVE PORTS

In this demonstration we look at the multi-mode WavePort. A WavePort solves the modes of its port face and
excites and absorbs each mode with its own propagation constant. The set of modes is fixed at the highest
frequency, so every S-matrix index belongs to the same physical mode at every frequency.

We use a 40 x 10 mm rectangular waveguide between 8 and 10 GHz. In this band two modes propagate:
TE10 (cutoff 3.75 GHz) and TE20 (cutoff 7.49 GHz). The next mode, TE30, cuts in at 11.2 GHz.

Part 1: A straight waveguide. Both modes are absorbed by the ports and do not couple into each other.
Part 2: The same waveguide with an asymmetrical metal post in the middle. The post couples the two modes.
        We look at the full 2-mode S-matrix.
"""
# [Written by Claude]
import numpy as np
import emerge as em
from emerge.plot import plot_sp

mm = 0.001
a, b = 40 * mm, 10 * mm   # waveguide width and height
L = 60 * mm               # waveguide length

############################################################
#           PART 1: A STRAIGHT TWO-MODE WAVEGUIDE          #
############################################################

with em.Simulation("MultimodeStraight") as model:
    model.check_version("3.0.7")  # Checks version compatibility

    # The waveguide runs along the Z-axis from z = 0 to z = L
    guide = em.geo.Box(a, b, L)

    model.commit_geometry()
    model.mw.set_frequency_range(8e9, 10e9, 11)
    model.mw.set_resolution(0.15)
    model.generate_mesh()
    model.view()

    # Each port includes two modes. The S-matrix index of port 1, mode 1 is 1.1, of port 1, mode 2 is 1.2, etc.
    # The modes are ordered by their propagation constant at the highest frequency: mode 1 = TE10, mode 2 = TE20.
    model.mw.bc.WavePort(guide.face("-z"), 1, number_of_modes=2)
    model.mw.bc.WavePort(guide.face("+z"), 2, number_of_modes=2)

    data = model.mw.run_sweep()
    g = data.scalar.grid

    # Both modes are transmitted with |S| = 1. The reflections and the conversion from one mode into the
    # other are many tens of dB down: the ports absorb each mode with its own propagation constant.
    plot_sp(g.freq, [g.S(2.1, 1.1), g.S(2.2, 1.2), g.S(1.1, 1.1), g.S(1.2, 1.2), g.S(1.2, 1.1), g.S(2.2, 1.1)],
            labels=["S(2.1,1.1) TE10 transmission", "S(2.2,1.2) TE20 transmission",
                    "S(1.1,1.1) TE10 reflection", "S(1.2,1.2) TE20 reflection",
                    "S(1.2,1.1) TE10 → TE20 reflected", "S(2.2,1.1) TE10 → TE20 transmitted"],
            dblim=[-100, 5])

    # Each port mode can be excited on its own. Here we show the TE20 mode of port 1.
    field = data.field.find(freq=9e9)
    field.excite_port(1.2)
    model.display.add_objects(*model.all_geos())
    model.display.animate().add_field(field.cutplane(1 * mm, y=b / 2).scalar("Ey", "complex"), symmetrize=True)
    model.display.show()

############################################################
#            PART 2: THE SAME GUIDE WITH A POST            #
############################################################

with em.Simulation("MultimodePost") as model:
    model.check_version("3.0.7")  # Checks version compatibility

    guide = em.geo.Box(a, b, L)

    # A metal post with a 2 mm radius stands on the bottom wall at a quarter of the width and fills two thirds
    # of the height. It is not symmetric, so it couples the TE10 and the TE20 mode.
    post = em.geo.Cylinder(2 * mm, 2 * b / 3, em.YAX.construct_cs((a / 4, 0, L / 2))).set_material(em.lib.PEC)

    model.commit_geometry()
    model.mw.set_frequency_range(8e9, 10e9, 11)
    model.mw.set_resolution(0.15)
    model.generate_mesh()
    model.view()

    model.mw.bc.WavePort(guide.face("-z"), 1, number_of_modes=2)
    model.mw.bc.WavePort(guide.face("+z"), 2, number_of_modes=2)

    data = model.mw.run_sweep()
    g = data.scalar.grid

    # The full 2-mode S-matrix: reflection, transmission and mode conversion for an incident TE10 and TE20 mode
    plot_sp(g.freq, [g.S(1.1, 1.1), g.S(2.1, 1.1), g.S(1.2, 1.1), g.S(2.2, 1.1)],
            labels=["S(1.1,1.1)", "S(2.1,1.1)", "S(1.2,1.1)", "S(2.2,1.1)"])
    plot_sp(g.freq, [g.S(1.2, 1.2), g.S(2.2, 1.2), g.S(1.1, 1.2), g.S(2.1, 1.2)],
            labels=["S(1.2,1.2)", "S(2.2,1.2)", "S(1.1,1.2)", "S(2.1,1.2)"])

    # The post is lossless, so all power that enters in one mode leaves in the two modes of the two ports:
    # every column of the 4 x 4 S-matrix has a total power of 1.
    for j in (1.1, 1.2, 2.1, 2.2):
        power = sum(np.abs(g.S(i, j).ravel()) ** 2 for i in (1.1, 1.2, 2.1, 2.2))
        print(f"Power out for an incident mode {j}: {power.min():.4f} ... {power.max():.4f}")

    # The field of an incident TE10 mode, partly converted into TE20 by the post
    field = data.field.find(freq=9e9)
    field.excite_port(1.1)
    model.display.add_objects(*model.all_geos())
    model.display.animate().add_field(field.cutplane(1 * mm, y=b / 2).scalar("Ey", "complex"), symmetrize=True)
    model.display.show()
