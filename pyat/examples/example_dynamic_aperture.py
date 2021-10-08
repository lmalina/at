import at
import at.physics.dynamic_aperture as da
import at.plot.dynamic_aperture as daplot
import at.lattice.cavity_access
import at.plot
import time
import numpy as np
import matplotlib.pyplot as plt

folder_data = '.'  # to store files and figures

sr_lattice_file = '/machfs/carver/pyat_dev/at/pyat/test_matlab/hmba.mat'
sr_lattice_variable = 'RING'


sr_arc = at.load_mat(sr_lattice_file, mat_key=sr_lattice_variable)

sr_ring = at.Lattice(sr_arc*32)

sr_ring.periodicity =1
sr_ring.set_rf_harmonic_number(992)
sr_ring.radiation_on()


DA = da.DynamicAperture(sr_ring, mode=['x','xp'], compute_limits=True, npoints=[11,11], grid_mode='grid', n_turns=10, dp=0.0)
#DA = da.DynamicAperture(sr_ring, mode=['x','y'], compute_limits=False, ranges=[[-1e-2, 1e-2], [-6e-3, 6e-3]], npoints=[13,13], grid_mode='radial', n_turns=250, dp=0.0)
DA.generate_particles()
DA.track_particles()
_, _, h, v, s = DA.compute_DA()

[print(h_, v_, s_) for h_, v_, s_ in zip(h, v, s)]
daplot.plot(DA, file_name_save=None)
plt.show()

