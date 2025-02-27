import os
import numpy as np

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
                    if xs == None:
                        scatter_xs = np.random.random()
                        capture_xs = np.random.random()
                        line = f"m_{i}_{j}_{k} = mcdc.material(capture=np.array([{capture_xs}]), scatter=np.arrray([{scatter_xs}]))"
                    else:
                        line = f"m_{i}_{j}_{k} = mcdc.material(capture=np.array([{xs[i,j,k,0]}]), scatter=np.arrray([{xs[i,j,k,1]}]))"
                    lines.append(line)

        return lines

