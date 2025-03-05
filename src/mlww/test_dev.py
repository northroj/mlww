import mlww.generate as generate
import numpy as np
import mlww.utility as utility
import matplotlib.pyplot as plt

"""
test1 = utility.test1()
test1.print1()
print(test1.c)
"""


"""

generate_input = generate.InputGeneration()
custom_materials = [
    "# Set materials",
    "m1 = mcdc.material(capture=np.array([1.0]))",
    "m2 = mcdc.material(capture=np.array([1.5]))",
    "m5 = mcdc.material(capture=np.array([2.0]))\n",  # Changed m3 to m5
]

generate_input.set_materials(lines=custom_materials)
generate_geom = generate.GeometryGeneration(x=2,y=1,z=5)
cell_lines = generate_geom.generate_cells()
surface_lines = generate_geom.generate_surfaces()
tally_lines = generate_geom.generate_tally()
mat_lines = generate_geom.generate_materials()
generate_input.set_surfaces(lines=surface_lines)
generate_input.set_cells(lines=cell_lines)
generate_input.set_tally(lines=tally_lines)
generate_input.set_materials(lines=mat_lines)
generate_input.write_to_file(filename="test1.py", directory="../../../mcdc_inputs")

"""

"""
generate_geom = generate.GeometryGeneration(x=2,y=1,z=5)
surface_lines = generate_geom.generate_surfaces()
cell_lines = generate_geom.generate_cells()
tally_lines = generate_geom.generate_tally()
print("\n".join(surface_lines))
print("\n".join(cell_lines))
print("\n".join(tally_lines))
"""

"""
generate_geom = generate.GeometryGeneration(2,2,5)
source_lines = generate_geom.generate_materials(np.zeros((2,2,5,3)))
print(source_lines)
"""



"""

random_generate = generate.RandomGeneration(10,10,5)
random_generate.randomize_structure(6)
random_grid = random_generate.get_grid()

#print(repr(random_grid))

plt.figure(2)
random_generate.plot_2d_grid(random_grid,0,0)
"""

"""
for i in range(5):
    random_generate = generate.RandomGeneration(2,3,i+1)
    random_generate.randomize_structure()
    random_generate.write_to_hdf5(filename="test_input_grid.h5", directory="../../../mcdc_inputs", wipe=True)
"""