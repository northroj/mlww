import mlww.train as train
import mlww.generate as generate
import numpy as np
import h5py
import os
import glob


def prepare_efficiency_comparison(randomize = True):
    if randomize:
        print("Randomizing input")
        # Generate a random geometry
        random_generate = generate.RandomGeneration(5,5,4)
        random_generate.randomize_structure()
        random_grid = random_generate.get_grid()
        random_generate.plot_2d_grid(random_grid)
        random_generate.write_to_hdf5(filename="efficiency_input.h5", directory="./", wipe=True)
    else:
        # I am manually setting an input configuration that would benefit from weight windows
        set_grid = generate.RandomGeneration(5,5,4)
        manual_grid = np.zeros((5,5,4,3))
        manual_grid[:1,:,:,2] = 0.5
        manual_grid[:2,:,:,0] = 0.1
        manual_grid[2:,:,:,0] = 0.1
        set_grid.set_grid(manual_grid)
        plot_grid = set_grid.get_grid()
        set_grid.plot_2d_grid(plot_grid)
        set_grid.write_to_hdf5(filename="efficiency_input.h5", directory="./", wipe=True)

    # create the mcdc input file
    with h5py.File("efficiency_input.h5", "r") as hdf5_file:
        grid_data = hdf5_file["case_0"][...]
        size_x, size_y, size_z, _ = grid_data.shape
        generate_input = generate.InputGeneration()
        generate_geom = generate.GeometryGeneration(x=size_x, y=size_y, z=size_z)
        cell_lines = generate_geom.generate_cells()
        surface_lines = generate_geom.generate_surfaces()
        tally_lines = generate_geom.generate_tally()
        mat_lines = generate_geom.generate_materials(grid_data)
        source_lines = generate_geom.generate_sources(grid_data)
        generate_input.set_surfaces(lines=surface_lines)
        generate_input.set_cells(lines=cell_lines)
        generate_input.set_tally(lines=tally_lines)
        generate_input.set_materials(lines=mat_lines)
        generate_input.set_source(lines=source_lines)
        generate_input.set_settings(N_particle=1e4)
        generate_input.write_to_file(filename=f"case_0.py", directory="./")

        # Also create the input using the generated weight windows
        loaded_model = train.ModelLoader("../models/mlww_5x5x4_030525.pt")
        predicted_flux = np.array(loaded_model.predict_flux(grid_data))
        generate_input.set_ww(predicted_flux)
        generate_input.write_to_file(filename="case_1.py", directory="./")

def run_efficiency_comparison():
    print("Running MC/DC")
    run_cases = generate.RunMcdc()
    run_cases.run_cases("./", use_numba = False, save_output=True)

def analyze_comparison():
    print("Calculating statistics")
    with h5py.File("case_0.h5", "r") as f:
        tally_mean = f["tallies"]["mesh_tally_0"]["flux"]["mean"][:]
        tally_std = f["tallies"]["mesh_tally_0"]["flux"]["sdev"][:]
        tally_std = np.divide(tally_std, tally_mean, out=np.zeros_like(tally_mean), where=tally_mean!=0)
        runtime = f["runtime"]["total"][:]
    with h5py.File("case_1.h5", "r") as f:
        tally_mean_ww = f["tallies"]["mesh_tally_0"]["flux"]["mean"][:]
        tally_std_ww = f["tallies"]["mesh_tally_0"]["flux"]["sdev"][:]
        tally_std_ww = np.divide(tally_std_ww, tally_mean_ww, out=np.zeros_like(tally_mean_ww), where=tally_mean_ww!=0)
        runtime_ww = f["runtime"]["total"][:]
    
    FOM = 1/(np.mean(tally_std*tally_std)*runtime)
    FOM_ww = 1/(np.mean(tally_std_ww*tally_std_ww)*runtime_ww)

    print("Average figure of merit without weight windows =", FOM)
    print("Average figure of merit with weight windows =", FOM_ww)
    loaded_model = train.ModelLoader("../models/mlww_5x5x4_030525.pt")
    loaded_model.plot_compare_flux(np.array(tally_mean), np.array(tally_mean_ww), z_slice=0)

    return tally_mean, tally_std, runtime, tally_mean_ww, tally_std_ww, runtime_ww

def clean_comparison():
    # Remove Python files starting with 'case_' and ending with '.py'
    for file in glob.glob("case_*.py"):
        os.remove(file)
        print(f"Deleted: {file}")
    
    # Remove all .h5 files
    for file in glob.glob("*.h5"):
        os.remove(file)
        print(f"Deleted: {file}")

prepare_efficiency_comparison(randomize=True)
run_efficiency_comparison()
tally_mean, tally_std, runtime, tally_mean_ww, tally_std_ww, runtime_ww = analyze_comparison()
clean_comparison()