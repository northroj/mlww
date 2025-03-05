import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import subprocess

class InputGeneration:
    def __init__(self):
        self.sections = {
            "materials": [],
            "surfaces": [],
            "cells": [],
            "source": [],
            "tally": [],
            "settings": []
        }

        self.lines = [
            "import numpy as np",
            "import mcdc\n",
        ]

        self.run_line = ["# Run", "mcdc.run()"]

        self.set_materials()
        self.set_surfaces()
        self.set_cells()
        self.set_source()
        self.set_tally()
        self.set_settings()
        self._add_run()

    def set_materials(self, lines=None):
        self.sections["materials"] = lines if lines else [
            "# Set materials",
            "m1 = mcdc.material(capture=np.array([1.0]))",
            "m2 = mcdc.material(capture=np.array([1.5]))",
            "m3 = mcdc.material(capture=np.array([2.0]))\n",
        ]
        self._update_lines()

    def set_surfaces(self, lines=None):
        self.sections["surfaces"] = lines if lines else [
            "# Set surfaces",
            "s1 = mcdc.surface(\"plane-z\", z=0.0, bc=\"vacuum\")",
            "s2 = mcdc.surface(\"plane-z\", z=2.0)",
            "s3 = mcdc.surface(\"plane-z\", z=4.0)",
            "s4 = mcdc.surface(\"plane-z\", z=6.0, bc=\"vacuum\")\n",
        ]
        self._update_lines()

    def set_cells(self, lines=None):
        self.sections["cells"] = lines if lines else [
            "# Set cells",
            "mcdc.cell(+s1 & -s2, m2)",
            "mcdc.cell(+s2 & -s3, m3)",
            "mcdc.cell(+s3 & -s4, m1)\n",
        ]
        self._update_lines()

    def set_source(self, lines=None):
        self.sections["source"] = lines if lines else [
            "# Set source",
            "mcdc.source(z=[0.0, 6.0], isotropic=True)\n",
        ]
        self._update_lines()

    def set_tally(self, lines=None):
        self.sections["tally"] = lines if lines else [
            "# Set tally",
            "mcdc.tally.mesh_tally(",
            "    scores=[\"flux\"],",
            "    z=np.linspace(0.0, 6.0, 61),",
            "    mu=np.linspace(-1.0, 1.0, 32 + 1),",
            ")\n",
        ]
        self._update_lines()

    def set_settings(self, N_particle=1e3):
        self.sections["settings"] = [
            "# Set settings",
            f"mcdc.setting(N_particle={N_particle})\n",
        ]
        self._update_lines()

    def _add_run(self):
        self.run_line = ["# Run", "mcdc.run()"]
        self._update_lines()

    def _update_lines(self):
        self.lines = [
            "import numpy as np",
            "import mcdc\n",
        ]
        for section in ["materials", "surfaces", "cells", "source", "tally", "settings"]:
            self.lines.extend(self.sections[section])
        self.lines.extend(self.run_line)

    def write_to_file(self, filename="mcdc_input.py", directory=None):
        if directory:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename

        with open(filepath, "w") as f:
            f.write("\n".join(self.lines))


class GeometryGeneration:
    def __init__(self, x=2, y=2, z=2):
        # Validate inputs
        for dim_name, dim_value in zip(['x', 'y', 'z'], [x, y, z]):
            if not isinstance(dim_value, int):
                raise ValueError(f"{dim_name} must be an integer.")
            if dim_value < 1 or dim_value > 100:
                raise ValueError(f"{dim_name} must be between 1 and 100.")
        
        self.x = x
        self.y = y
        self.z = z

    def generate_surfaces(self):
        lines = ["# Set surfaces"]

        # Helper function to generate plane definitions
        def generate_planes(axis, max_value):
            for i in range(max_value + 1):
                bc = 'bc="vacuum"' if i == 0 or i == max_value else ''
                coord = f"{axis}={i}.0"
                line = f"s_{axis}_{i} = mcdc.surface(\"plane-{axis}\", {coord}{', ' + bc if bc else ''})"
                lines.append(line)

        # Generate planes for x, y, and z axes
        generate_planes('x', self.x)
        generate_planes('y', self.y)
        generate_planes('z', self.z)

        return lines
    
    def generate_cells(self):
        lines = ["# Set cells"]

        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    cell_surfaces = (
                        f"+s_x_{i} & -s_x_{i+1} & "
                        f"+s_y_{j} & -s_y_{j+1} & "
                        f"+s_z_{k} & -s_z_{k+1}"
                    )
                    material = f"m_{i}_{j}_{k}"
                    line = f"mcdc.cell({cell_surfaces}, {material})"
                    lines.append(line)

        return lines
    
    def generate_tally(self):
        lines = [
            "# Set tally",
            "mcdc.tally.mesh_tally(",
            "    scores=[\"flux\"],",
            f"    x=np.linspace(0.0,{self.x},{self.x+1}),",
            f"    y=np.linspace(0.0,{self.y},{self.y+1}),",
            f"    z=np.linspace(0.0,{self.z},{self.z+1}),",
            ")\n"
        ]

        return lines
    
    def generate_materials(self, xs=None):
        lines = ["# Set materials"]

        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    if xs is None:
                        scatter_xs = np.random.random()
                        capture_xs = np.random.random()
                        line = f"m_{i}_{j}_{k} = mcdc.material(capture=np.array([{capture_xs}]), scatter=np.array([[{scatter_xs}]]))"
                    else:
                        line = f"m_{i}_{j}_{k} = mcdc.material(capture=np.array([{xs[i,j,k,0]}]), scatter=np.array([[{xs[i,j,k,1]}]]))"
                    lines.append(line)

        return lines
    
    def generate_sources(self, xs=None):
        lines = ["# Set sources"]

        for i in range(self.x):
            for j in range(self.y):
                for k in range(self.z):
                    if xs is None:
                        random_source_strength = np.random.random()
                        line = f"mcdc.source(x=[{i}, {i+1}], y=[{j},{j+1}], z=[{k},{k+1}], isotropic=True, prob={random_source_strength})"
                    else:
                        line = f"mcdc.source(x=[{i}, {i+1}], y=[{j},{j+1}], z=[{k},{k+1}], isotropic=True, prob={xs[i,j,k,2]})"
                    lines.append(line)

        return lines


class RandomGeneration:
    def __init__(self, size_x, size_y, size_z):
        """
        Initialize the 4D array with random values.
        Dimensions: (x, y, z, data_type), where data_type corresponds to:
        0 - Capture Cross Section
        1 - Scattering Cross Section
        2 - Source Strength
        """
        self.size = (size_x, size_y, size_z)
        
        # Initialize random values
        #self.grid = np.random.rand(size_x, size_y, size_z, 3)
        self.grid = np.zeros((size_x, size_y, size_z, 3))
    
    def randomize_structure(self, seed=None):
        """
        Randomly generates a structured pattern by placing and expanding blobs.
        """
        if seed is not None:
            np.random.seed(seed)
        total_cells = self.size[0] * self.size[1] * self.size[2]
        num_blobs = np.random.randint(1, max(2, total_cells // 8))
        
        for _ in range(num_blobs):
            self.generate_blob()
    
    def generate_blob(self):
        """
        Generates a single blob with random properties and expands it randomly.
        """
        # Generate random properties for the blob
        blob_values = np.random.rand(3)  # Capture, Scatter, Source
        
        # Pick a random starting location
        x, y, z = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]), np.random.randint(0, self.size[2])
        self.grid[x, y, z] = blob_values
        
        # Determine the number of steps
        total_cells = self.size[0] * self.size[1] * self.size[2]
        num_steps = np.random.randint(1, max(2, total_cells // 8))
        
        for _ in range(num_steps):
            out_of_range = True
            while out_of_range:
                # Pick a random direction: -1 or +1 in x, y, or z
                direction = np.random.choice(['x', 'y', 'z'])
                step = np.random.choice([-1, 1])
                
                new_x, new_y, new_z = x, y, z
                if direction == 'x':
                    new_x += step
                elif direction == 'y':
                    new_y += step
                else:
                    new_z += step
                
                # Ensure the step remains within bounds
                if 0 <= new_x < self.size[0] and 0 <= new_y < self.size[1] and 0 <= new_z < self.size[2]:
                    x, y, z = new_x, new_y, new_z
                    self.grid[x, y, z] = blob_values
                    out_of_range = False

    
    def get_grid(self):
        """
        Return the generated grid.
        """
        return self.grid
    
    def plot_2d_grid(self, data_grid, data_type=0, z_slice=0):
        """
        Plots a 2D heatmap of the grid at z=0 for the specified data type.
        data_type: 0 (Capture Cross Section), 1 (Scattering Cross Section), or 2 (Source Strength)
        """
        if data_type not in [0, 1, 2]:
            raise ValueError("data_type must be 0, 1, or 2")
        if z_slice > self.size[2]:
            raise ValueError("Z value must be in the grid dimension (0 indexed)")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(data_grid[:, :, z_slice, data_type], cmap='viridis', origin='lower', extent=[0, self.size[0], 0, self.size[1]])
        plt.colorbar(label=['Capture Cross Section', 'Scattering Cross Section', 'Source Strength'][data_type])
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.title(f"2D Grid Visualization - {['Capture Cross Section', 'Scattering Cross Section', 'Source Strength'][data_type]}")
        plt.show()
    
    def write_to_hdf5(self, filename, directory=None, wipe=False):
        """
        Writes the 4D grid to an HDF5 file.
        If wipe is True, it clears the file before adding the new input.
        If wipe is False, it appends the new input under the lowest available case number.
        """
        mode = 'w' if wipe else 'a'

        if directory:
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename
        
        with h5py.File(filepath, mode) as hdf5_file:
            if wipe:
                case_number = 0
            else:
                existing_cases = [int(key.split('_')[1]) for key in hdf5_file.keys() if key.startswith("case_")]
                case_number = min(set(range(len(existing_cases) + 1)) - set(existing_cases), default=0)
            
            case_name = f"case_{case_number}"
            hdf5_file.create_dataset(case_name, data=self.grid)



class RunMcdc:
    def run_cases(self, directory, start=0, end=None, use_numba=True, tally_filename="tally_results.h5"):
        """
        Runs the case_#.py files in the specified directory within the given range.

        :param directory: Path to the directory containing case_#.py files.
        :param start: The starting case number.
        :param end: The ending case number. If None, it is set to the largest case number found in the directory.
        :param use_numba: If True, passes "--mode=numba" as an argument when running the script.
        :param tally_filename: Name of the HDF5 file where tally results will be stored.
        """
        if end is None:
            case_numbers = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(directory) if f.startswith("case_") and f.endswith(".py")]
            end = max(case_numbers) if case_numbers else start
        
        args = ["--mode=numba"] if use_numba else []
        tally_path = os.path.join(directory, tally_filename)
        
        for case_num in range(start, end + 1):
            case_file = f"case_{case_num}.py"
            if os.path.isfile(os.path.join(directory, case_file)):
                print(f"Running {case_file} with arguments: {args} in directory {directory}")
                subprocess.run(["python", case_file] + args, check=True, cwd=directory)
                self.collect_tally(directory, case_num, tally_path)
            else:
                print(f"Warning: {case_file} not found.")
    
    def collect_tally(self, directory: str, case_num: int, tally_path: str):
        """
        Collects the tally data from output.h5 and stores it in the tally HDF5 file.

        :param directory: Path to the directory containing the output.h5 file.
        :param case_num: The case number for which the tally is being collected.
        :param tally_path: Path to the HDF5 file where tally results will be stored.
        """
        output_file = os.path.join(directory, "output.h5")
        if not os.path.isfile(output_file):
            print(f"Warning: {output_file} not found for case {case_num}.")
            return
        
        with h5py.File(output_file, "r") as f:
            try:
                tally_data = f["tallies"]["mesh_tally_0"]["flux"]["mean"][:]
            except KeyError as e:
                print(f"Error: Could not find expected dataset in {output_file} for case {case_num}: {e}")
                return
        
        with h5py.File(tally_path, "a") as f:
            dataset_name = f"case_{case_num}"
            if dataset_name in f:
                del f[dataset_name]  # Delete existing dataset to overwrite
            f.create_dataset(dataset_name, data=tally_data)
            print(f"Stored tally for {dataset_name} in {tally_path}.")



