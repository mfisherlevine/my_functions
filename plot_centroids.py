import numpy as np
import pylab as pl

# filename = '/mnt/hgfs/VMShared/Data/lsst_calibration/4/centroids.txt'
filename = '/mnt/hgfs/VMShared/Data/lsst_calibration/1/centroids.txt'


data = np.loadtxt(filename, dtype=str, skiprows=1, usecols=[1,3,4], unpack=False)
fig = pl.figure(figsize=[14,10])
ax = fig.add_subplot(111)

i = 1
for text,x,y in zip(data[:,0],data[:,1],data[:,2]):
    if textt == 'nominal':
        pl.plot(x,y,'ro')
        ax.annotate(str(i), xy=[x,y])
    else:
        pl.plot(x,y,'bo')
        ax.annotate(text + ', ' + str(i), xy=[x,y])
    i += 1

ylims = ax.get_ylim()
ax.set_ylim([ylims[0],ylims[1]+0.5])

pl.grid()
pl.tight_layout()
pl.savefig('/mnt/hgfs/VMShared/Data/lsst_calibration/1/centroids.pdf')
pl.savefig('/mnt/hgfs/VMShared/Data/lsst_calibration/1/centroids.png')
pl.show()
print 'Finished'