# This python file is an example used to generate the training data for the default model (static 5x5x4 grid)

import mlww.generate as generate
import numpy as np
import h5py

def add_to_training_data(n_it, filename = "input_parameters.h5", directory = "../../mcdc_inputs", clear_first = False):
    for case in range(n_it):
        random_generate = generate.RandomGeneration(5,5,4)
        random_generate.randomize_structure()
        if clear_first == True and case == 0:
            random_generate.write_to_hdf5(filename=filename, directory=directory, wipe=True)
        else:
            random_generate.write_to_hdf5(filename=filename, directory=directory, wipe=False)

def convert_to_mcdc_input(filepath, i_start=0, i_end=None):
    with h5py.File(filepath, "r") as hdf5_file:
        # Determine the maximum case number if i_end is not provided
        existing_cases = [int(key.split('_')[1]) for key in hdf5_file.keys() if key.startswith("case_")]
        if i_end is None:
            i_end = max(existing_cases) if existing_cases else 0
        
        # Loop through each case
        for case in range(i_start, i_end + 1):
            case_key = f"case_{case}"
            if case_key not in hdf5_file:
                continue  # Skip missing cases
            
            grid_data = hdf5_file[case_key][...]
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
            generate_input.set_settings(N_particle=1e5)
            
            generate_input.write_to_file(filename=f"case_{case}.py", directory="../../mcdc_inputs")

def run_mcdc_cases(directory, start=0):
    run_cases = generate.RunMcdc()
    run_cases.run_cases(directory, start = start, use_numba = True)

add_to_training_data(10, clear_first = True)
convert_to_mcdc_input("../../mcdc_inputs/input_parameters.h5")
run_mcdc_cases("../../mcdc_inputs/", start=1803)