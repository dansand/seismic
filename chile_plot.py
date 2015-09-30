import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.basemap import Basemap
import obspy
from obspy.imaging.beachball import beach
import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from matplotlib import cm, rcParams

#Chile quake
catalog = obspy.read_events('./data/SPUD_NDK_bundle_2015-09-21T07.43.47.txt')
filtered = (
    catalog.filter('magnitude >= 8.0', 'magnitude <= 9',
                   'time > 2005-05-01T00:00')
)

clon = np.floor(filtered[0]['origins'][0]['longitude'])
clat = np.floor(filtered[0]['origins'][0]['latitude'])


m = Basemap(projection='cyl',llcrnrlat=clat - 5,urcrnrlat= clat + 5,\
            llcrnrlon= clon-5,urcrnrlon=clon+5,resolution='l')

plt.figure(figsize=(8., 8.))
ax = plt.subplot(111)

###########
#Earthquakes
###########
for event in filtered:
    fc = event.focal_mechanisms[0]
    plane = fc.nodal_planes.nodal_plane_1
    origin = event.origins[0]
    depth = origin
    bb = beach([plane.strike, plane.dip, plane.rake], xy=m(origin.longitude, origin.latitude),
               width=0.6, linewidth=0.5, facecolor='b')
    ax.add_collection(bb)

m.fillcontinents(alpha=0.5)

m.drawparallels(np.arange(-90.,91.,10.), labels=[False,True,True,False])
m.drawmeridians(np.arange(-180.,181.,10.), labels=[True,False,False,True])
#m.etopo(alpha=0.5)
m.drawcoastlines(linewidth=0.5)
m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1500, verbose= True)
plt.title("Magnitude 8.3 Offshore Coquimbo, Chile")
plt.show()
