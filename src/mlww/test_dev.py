import mlww.generate as generate

import mlww.utility as utility

"""
test1 = utility.test1()
test1.print1()
print(test1.c)
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
generate_geom = generate.GeometryGeneration(x=2,y=1,z=5)
surface_lines = generate_geom.generate_surfaces()
cell_lines = generate_geom.generate_cells()
tally_lines = generate_geom.generate_tally()
print("\n".join(surface_lines))
print("\n".join(cell_lines))
print("\n".join(tally_lines))
"""