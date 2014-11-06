import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

    
def HistogramImageData(np.ndarray[DTYPE_t, ndim=2] data, int power_threshold):
    assert data.dtype == DTYPE
    
    cdef int histmin, histmax, nbins, xsize, ysize, n_power_points, npts
    cdef double value, power, total_power
    
    from ROOT import TH1F
    
    histmin = -100
    histmax = 150000    # approx fullwell
    nbins = int(histmax - histmin + 1)/5      # binsize of 1
    
    image_hist = TH1F('', '',nbins,histmin,histmax)

#    xsize, ysize = data.shape # will not cythonize :/

    xsize = len(data) 
    ysize = len(data[1])
    
    npts = 0
    power = 0
    n_power_points = 0 
    total_power = 0.0
    
    for x in range(xsize):
        for y in range(ysize):
            value = data[x,y]
            image_hist.Fill(value)
            total_power += value
            if value >= float(power_threshold):
                power += value
                n_power_points += 1
#            npts += 1
                
#    print "%s points looped over"%npts
#    print "Power above thr    = %s"%power
#    print "Total power        = %s"%total_power
    
    return image_hist, power, n_power_points





