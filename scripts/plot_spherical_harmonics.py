from mayavi import mlab
import numpy as np
from scipy.special import sph_harm

# Create a sphere
r = 0.4
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:201j, 0:2 * pi:201j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()

# Represent spherical harmonics on the surface of the sphere
for n in range(0, 4):
    s_tot = np.zeros((201, 201))
    for m in range(-n, 1):
        s = np.abs(sph_harm(m, n, theta, phi).real)
        s_tot = s_tot + s
        s = s_tot

        # mlab.mesh(x - m, y - n, z, scalars=s, colormap='jet')
        mlab.mesh(x - m, y - n, z, scalars=s, colormap='Spectral')

        s[s < 0] *= 0.97

        s /= s.max()
        mlab.mesh(s * x - m, s * y - n, s * z + 1.3, scalars=s, colormap='Spectral')  # scalars=s, colormap='jet')

mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
mlab.show()
