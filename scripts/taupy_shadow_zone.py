import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel

# set the velocity model you want to use:
model = TauPyModel(model='iasp91')

# set epicentral distances, the phases you want, and a source depth:
DISTANCES=range(0,360,5)
PHASES = ['P','PKP']
sourcedepth=50

# ax_right is used for paths plotted on the right half.
fig, ax_right = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax_right.set_theta_zero_location('N')
ax_right.set_theta_direction(-1)
# ax_left is used for paths plotted on the left half.
ax_left = fig.add_axes(ax_right.get_position(), projection='polar',
                       label='twin', frameon=False)
ax_left.set_theta_zero_location('N')
ax_left.set_theta_direction(+1)
ax_left.xaxis.set_visible(False)
ax_left.yaxis.set_visible(False)

# Plot all ray paths:
for distance in DISTANCES:
    if distance < 0:
        realdist = -distance
        ax = ax_left
    else:
        realdist = distance
        ax = ax_right

    arrivals = model.get_ray_paths(sourcedepth, realdist, phase_list=PHASES)
    if not len(arrivals):
        #print('FAIL', PHASES, distance)
        continue
    arrivals.plot(plot_type='spherical',
                  legend=False, label_arrivals=True,
                  show=False, ax=ax)

# Annotate regions:
ax_right.text(0, 0, 'Solid\ninner\ncore',
              horizontalalignment='center', verticalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
ocr = (model.model.radiusOfEarth -
       (model.model.sMod.vMod.iocbDepth + model.model.sMod.vMod.cmbDepth) / 2)
ax_right.text(np.deg2rad(180), ocr, 'Fluid outer core',
              horizontalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
mr = model.model.radiusOfEarth - model.model.sMod.vMod.cmbDepth / 2
ax_right.text(np.deg2rad(180), mr, 'Solid mantle',
              horizontalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.show()
