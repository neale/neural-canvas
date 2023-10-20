import numpy as np    
import pyvista as pv


values = np.random.randint(0, 255, size=(100, 100, 100), dtype=np.uint8)
grid = pv.UniformGrid(dimensions=values.shape)
p = pv.Plotter()
vol = p.add_volume(
	grid,
	scalars=values.reshape(-1),
	mapper='open_gl')

fp = "./test_export"

#p.export_vtksz(fp)
#p.export_gltf(fp)
#p.export_vrml(fp)
p.export_html(fp)
p.show()
