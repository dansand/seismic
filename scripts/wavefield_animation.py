import PIL.Image
import matplotlib.pylab as plt
from scipy import misc
from fatiando import utils
import numpy as np
from matplotlib import animation
from fatiando import gridder
from fatiando.seismic import wavefd
from fatiando.vis import mpl
import urllib

plt.switch_backend("nbagg")

urllib.urlretrieve(
                   'http://www.reproducibility.org/RSF/book/cwp/geo2009TTIModeSeparation/marmousi2/Fig/vp.png', 'image.png')

image = misc.fromimage(PIL.Image.open('image.png'), flatten=True)

im2 = image[170:380,70:720]

def scalearray(fname, ranges=None, shape=None):
    #image = scipy.misc.fromimage(PIL.Image.open(fname), flatten=True)
    # Invert the color scale and normalize
    values = (fname.max() - fname)/np.abs(fname).max()
    if ranges is not None:
        vmin, vmax = ranges
        values *= vmax - vmin
        values += vmin
    if shape is not None and tuple(shape) != values.shape:
        ny, nx = values.shape
        X, Y = np.meshgrid(range(nx), range(ny))
        values = gridder.interp(X.ravel(), Y.ravel(), values.ravel(),
                                shape)[2].reshape(shape)
    return values

fim = scalearray(im2, ranges=[4500, 1500], shape=None)




shape = (210, 650)
area = [0, 65000*10, 0, 21000*10]

fim[:,350:] = 4500


# Make a density and wave velocity model
#density = 2400*np.ones(shape)
pvel = fim
svel = pvel*0.6
density = pvel*0.6
density = np.float64(density)
moho = 40
density[moho:] = 3300
svel[moho:] = 4300
pvel[moho:] = 7500
mu = wavefd.lame_mu(svel, density)
lamb = wavefd.lame_lamb(pvel, svel, density)

# Make a wave source from a mexican hat wavelet for the x and z directions
sources = [
           [wavefd.MexHatSource(350000, 20000, area, shape, 100000, 0.5, delay=2)],
           [wavefd.MexHatSource(350000, 20000, area, shape, 100000, 0.5, delay=2)]]

# Get the iterator. This part only generates an iterator object. The actual
# computations take place at each iteration in the for loop below
dt = wavefd.maxdt(area, shape, pvel.max())
duration = 100
maxit = int(duration/dt)
stations = [[550000, 0]]
snapshots = int(1./dt)
simulation = wavefd.elastic_psv(lamb, mu, density, area, dt, maxit, sources,
                                stations, snapshots, padding=70, taper=0.005, xz2ps=True)


# This part makes an animation using matplotlibs animation API
background = 10**-5*((pvel - pvel.min())/pvel.max())
fig = mpl.figure(figsize=(10, 8))
mpl.subplots_adjust(right=0.98, left=0.11, hspace=0.3, top=0.93)
mpl.subplot(2, 1, 1)
mpl.title('x seismogram')
xseismogram, = mpl.plot([],[],'-k')
mpl.xlim(0, duration)
mpl.ylim(-0.05, 0.05)
mpl.ylabel('Amplitude')
ax = mpl.subplot(2, 1, 2)
mpl.title('time: 0.0 s')
wavefield = mpl.imshow(density, extent=area, cmap=mpl.cm.gray_r,
                       vmin=-0.00001, vmax=0.00001)
mpl.points(stations, '^b', size=8)
#mpl.text(500000, 20000, 'Crust')
#mpl.text(500000, 60000, 'Mantle')
fig.text(0.8, 0.5, 'Seismometer')
mpl.xlim(area[:2])
mpl.ylim(area[2:][::-1])
mpl.xlabel('x (km)')
mpl.ylabel('z (km)')
mpl.m2km()
times = np.linspace(0, dt*maxit, maxit)
# This function updates the plot every few timesteps
def animate(i):
    t, p, s, xcomp, zcomp = simulation.next()
    mpl.title('time: %0.1f s' % (times[t]))
    wavefield.set_array((background + p + s)[::-1])
    xseismogram.set_data(times[:t+1], xcomp[0][:t+1])
    return wavefield, xseismogram
anim = animation.FuncAnimation(fig, animate, frames=maxit/snapshots, interval=1)
mpl.show()
