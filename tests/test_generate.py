import pytest
import mlww.generate as generate

# integration test for mcdc input generation
def test_generate_input():
    generate_input = generate.InputGeneration()
    generate_geom = generate.GeometryGeneration(x=2,y=1,z=5)
    cell_lines = generate_geom.generate_cells()
    surface_lines = generate_geom.generate_surfaces()
    tally_lines = generate_geom.generate_tally()
    generate_input.set_surfaces(lines=surface_lines)
    generate_input.set_cells(lines=cell_lines)
    generate_input.set_tally(lines=tally_lines)

    total_lines = generate_input.lines
    assert total_lines == ['import numpy as np', 'import mcdc\n', '# Set materials', 'm1 = mcdc.material(capture=np.array([1.0]))', 'm2 = mcdc.material(capture=np.array([1.5]))', 'm3 = mcdc.material(capture=np.array([2.0]))\n', '# Set surfaces', 's_x_0 = mcdc.surface("plane-x", x=0.0, bc="vacuum")', 's_x_1 = mcdc.surface("plane-x", x=1.0)', 's_x_2 = mcdc.surface("plane-x", x=2.0, bc="vacuum")', 's_y_0 = mcdc.surface("plane-y", y=0.0, bc="vacuum")', 's_y_1 = mcdc.surface("plane-y", y=1.0, bc="vacuum")', 's_z_0 = mcdc.surface("plane-z", z=0.0, bc="vacuum")', 's_z_1 = mcdc.surface("plane-z", z=1.0)', 's_z_2 = mcdc.surface("plane-z", z=2.0)', 's_z_3 = mcdc.surface("plane-z", z=3.0)', 's_z_4 = mcdc.surface("plane-z", z=4.0)', 's_z_5 = mcdc.surface("plane-z", z=5.0, bc="vacuum")', '# Set cells', 'mcdc.cell(+s_x_0 & -s_x_1 & +s_y_0 & -s_y_1 & +s_z_0 & -s_z_1, m_0_0_0)', 'mcdc.cell(+s_x_0 & -s_x_1 & +s_y_0 & -s_y_1 & +s_z_1 & -s_z_2, m_0_0_1)', 'mcdc.cell(+s_x_0 & -s_x_1 & +s_y_0 & -s_y_1 & +s_z_2 & -s_z_3, m_0_0_2)', 'mcdc.cell(+s_x_0 & -s_x_1 & +s_y_0 & -s_y_1 & +s_z_3 & -s_z_4, m_0_0_3)', 'mcdc.cell(+s_x_0 & -s_x_1 & +s_y_0 & -s_y_1 & +s_z_4 & -s_z_5, m_0_0_4)', 'mcdc.cell(+s_x_1 & -s_x_2 & +s_y_0 & -s_y_1 & +s_z_0 & -s_z_1, m_1_0_0)', 'mcdc.cell(+s_x_1 & -s_x_2 & +s_y_0 & -s_y_1 & +s_z_1 & -s_z_2, m_1_0_1)', 'mcdc.cell(+s_x_1 & -s_x_2 & +s_y_0 & -s_y_1 & +s_z_2 & -s_z_3, m_1_0_2)', 'mcdc.cell(+s_x_1 & -s_x_2 & +s_y_0 & -s_y_1 & +s_z_3 & -s_z_4, m_1_0_3)', 'mcdc.cell(+s_x_1 & -s_x_2 & +s_y_0 & -s_y_1 & +s_z_4 & -s_z_5, m_1_0_4)', '# Set source', 'mcdc.source(z=[0.0, 6.0], isotropic=True)\n', '# Set tally', 'mcdc.tally.mesh_tally(', '    scores=["flux"],', '    x=np.linspace(0.0,2,3),', '    y=np.linspace(0.0,1,2),', '    z=np.linspace(0.0,5,6),', ')\n', '# Set settings', 'mcdc.setting(N_particle=1000.0)\n', '# Run', 'mcdc.run()']

