import emerge as em

""" DIELECTRIC RESONATOR FILTER DEMO

This demo constructs and analyzes a dielectric resonator filter using EMerge.
A dielectric cylinder (resonator) sits on a supporting block within a metal
enclosure. We solve for its resonant modes via eigenmode analysis and visualize
electric and magnetic field distributions for each mode. """

# --- Unit definitions -----------------------------------------------------
mm = 0.001                       # meter per millimeter
inch = 25.4 * mm                 # meter per inch

# --- Geometry dimensions --------------------------------------------------
S = 2.03 * inch                  # enclosure height
W = 2.0 * inch                   # enclosure width/length (square base)

# Supporting block (substrate) dimensions
Dsup = 0.56 * inch               # support cylinder diameter
Lsup = 0.8 * inch                # support cylinder height
# Resonator cylinder dimensions
Dres = 1.176 * inch              # resonator cylinder diameter
Lres = 0.481 * inch              # resonator cylinder height

# --- Material definitions ------------------------------------------------
# High-er support material (e.g., alumina)
mat_support = em.lib.Material(er=10, color="#ffffff")
# Dielectric resonator material (e.g., ceramic)
mat_resonator = em.lib.Material(er=34, color="#ededed")

# Number of resonant modes to extract
Nmodes = 5

# --- Create simulation ---------------------------------------------------
model = em.Simulation('DielectricResonatorFilter')
model.check_version("0.6.7") # Checks version compatibility.

# --- Build geometry ------------------------------------------------------
# Metal enclosure box (PEC by default)
box = em.geo.Box(
    W, W, S,
    position=(-W/2, -W/2, 0)
)
# Support cylinder centered on enclosure floor
support = em.geo.Cylinder(
    radius=Dsup/2,
    height=Lsup,
    cs=em.GCS,
    Nsections=20
).set_material(mat_support).prio_up()
# DDR cylinder placed atop support
resonator = em.geo.Cylinder(
    radius=Dres/2,
    height=Lres,
    cs=em.GCS.displace(0, 0, Lsup),
    Nsections=32
).set_material(mat_resonator).prio_up()

# Assemble geometry into model
model.commit_geometry()

# --- Solver settings -----------------------------------------------------
# Only eigenmode solver needed; set center frequency estimate
model.mw.set_frequency(3e9)

# --- Mesh generation -----------------------------------------------------
model.generate_mesh()

# --- Eigenmode analysis --------------------------------------------------
# Solve for the first Nmodes resonant frequencies around 2 GHz
data = model.mw.eigenmode(
    2e9,
    nmodes=Nmodes
)

# --- Visualization of modes ----------------------------------------------
for mode_index in range(Nmodes):
    # Extract field grid for this mode (sample spacing ~0.2 in)
    field = data.field[mode_index].grid(0.2 * inch)
    # Show enclosure, support, and resonator transparently
    model.display.add_object(box, opacity=0.1)
    model.display.add_object(support, opacity=0.3)
    model.display.add_object(resonator, opacity=0.3)
    # Plot E-field vectors in red and H-field vectors in blue
    Evec = field.vector('E', 'real')
    Hvec = field.vector('H', 'real')
    model.display.add_quiver(*Evec, color='red')
    model.display.add_quiver(*Hvec, color='blue')
    # Annotate resonant frequency and field labels
    freq_ghz = data.field[mode_index].freq.real / 1e9
    model.display.add_title(f'Mode {mode_index+1}: {freq_ghz:.3f} GHz')
    model.display.add_text('E-field', color='red', abs_position=(0, 0.95))
    model.display.add_text('H-field', color='blue', abs_position=(0, 0.9))
    model.display.add_surf(*data.field[mode_index].cutplane(2*mm, y=0).scalar('normS','real'))
    # Render each mode one at a time
    model.display.show()
