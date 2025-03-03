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





def plot_2d_grid(data_grid, data_type=0):
    """
    Plots a 2D heatmap of the grid at z=0 for the specified data type.
    data_type: 0 (Capture Cross Section), 1 (Scattering Cross Section), or 2 (Source Strength)
    """
    if data_type not in [0, 1, 2]:
        raise ValueError("data_type must be 0, 1, or 2")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(data_grid[:, :, 0, data_type], cmap='viridis', origin='lower', extent=[0, 10, 0, 10])
    plt.colorbar(label=['Capture Cross Section', 'Scattering Cross Section', 'Source Strength'][data_type])
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title(f"2D Grid Visualization - {['Capture Cross Section', 'Scattering Cross Section', 'Source Strength'][data_type]}")
    plt.show()

random_generate = generate.RandomGeneration(10,10,1)
random_generate.randomize_structure()
random_grid = random_generate.get_grid()
plt.figure(1)
plot_2d_grid(random_grid,0)