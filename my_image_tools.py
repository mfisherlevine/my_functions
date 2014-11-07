def HistogramImage_NoAssembly(filename, power_threshold = -100000):
    from ROOT import TH1F
    import lsst.afw.image as afwImg
    
    histmin = 0
    histmax = 150000 # approx fullwell
    nbins = 150001 #binsize of 1
    image_hist = TH1F('', '',nbins,histmin,histmax)
    
    imagelist = [afwImg.DecoratedImageI(filename, i + 2) for i in range(16)]
    
    panelsize_x = imagelist[1].getWidth()
    panelsize_y = imagelist[1].getHeight()
    
    xsize = 8 * panelsize_x
    ysize = 2 * panelsize_y

    ypanels = 2
    xpanels = 8
    
    npts = 0
    power = 0.
    n_power_points = 0
   
    for panely in range(ypanels):
        for panelx in range(xpanels):
            panelnum = (panely*xpanels) + panelx
            for pixely in range(panelsize_y):
                for pixelx in range(panelsize_x):
                    value = imagelist[panelnum].getImage().get(pixelx,pixely)
                    image_hist.Fill(value)
                    if value >= power_threshold:
                        power += value
                        n_power_points += 1
                    npts += 1
                    
    print "Filled %s points"%npts
    return image_hist, power, n_power_points



def HistogramImage(filename, power_threshold = -100000):
    from ROOT import TH1F
    import lsst.afw.image as afwImg
    import numpy as np
    
    histmin = 0
    histmax = 150000 # approx fullwell
    nbins = 150001 #binsize of 1
    image_hist = TH1F('', '',nbins,histmin,histmax)
    
    from image_assembly import AssembleImage
    metadata_filename = '/home/mmmerlin/useful/herring_bone.fits'
    image = AssembleImage(filename, metadata_filename, False)

    data = image.getArray()
    xsize, ysize = data.shape

#    print xsize
#    print ysize
#    xsize = 4004
#    ysize = 4096 
    
    npts = 0
    power = 0.
    n_power_points = 0
    total_power = 0.
   
    print "wrong total power  = ",data.sum() #wrong
    print "right power        = ",data.astype('f8').sum() #right
    print "right power        = ",data.sum(dtype = 'f8')
    
    
    w = np.where(data >= power_threshold)
    print "power above thr    = ", data[w].sum(dtype = 'f8')

   
    for x in range(xsize):
        for y in range(ysize):
            value = image.get(y,x)
#            value = int(data[x,y])
            image_hist.Fill(value)
            total_power += value
            
            if value >= power_threshold:
                power += value
                n_power_points += 1
            npts += 1
                    
    print "Power above thr    = %s"%power
    print "Total power        = %s"%total_power
    
    return image_hist, power, n_power_points


def FastHistogramImage(filename, power_threshold = -100000):
    from image_assembly import AssembleImage
    from myCythonTools import HistogramImageData
    
    metadata_filename = '/home/mmmerlin/useful/herring_bone.fits'
    image = AssembleImage(filename, metadata_filename, False)
    
    power = float(0.)
    n_power_points = int(0)
    power_threshold_int = int(power_threshold)
   
    hist, power, n_power_points = HistogramImageData(image.getArray(), power_threshold_int)
    
    
    return hist, power, n_power_points

def FastHistogramImageData(data, power_threshold = -100000):
    from myCythonTools import HistogramImageData
    
    power = float(0.)
    n_power_points = int(0)
    power_threshold_int = int(power_threshold)
   
    hist, power, n_power_points = HistogramImageData(data, power_threshold_int)
    
    return hist, power, n_power_points



def FastHistogramImageData_Print(data, filename, histmin = -999999999, histmax = 999999999, fit_gaus = False):
    from myCythonTools import HistogramImageData
    from ROOT import TCanvas
    
    power = float(0.)
    n_power_points = int(0)
    power_threshold_int = int(0)
    
    c1 = TCanvas('c','c',1600,800)
    hist, power, n_power_points = HistogramImageData(data, power_threshold_int)
    hist.Draw()
    if histmin != -999999999 and histmax != 999999999:
        hist.GetXaxis().SetRangeUser(histmin,histmax)
    if fit_gaus: hist.Fit('gaus')
    c1.SaveAs(filename)
    
    return hist, power, n_power_points

def GetADC_OffsetsAndNoisesFromBiasFiles(path):
    import lsst.afw.math as math
    import numpy
    from image_assembly import GetImage_SingleAmp
    
    ADU_Values = numpy.ndarray(16,'f8')
    Noise_factors = numpy.ndarray(16,'f8')
    bias_files = [path + filename for filename in os.listdir(path) if filename.find('bias') != -1]
    
    for thisfile in bias_files:
        for amp in range(16):
            image = GetImage_SingleAmp(thisfile,False,amp)

            statFlags = math.MEAN | math.STDEV | math.MAX | math.MIN | math.STDEVCLIP | math.MEANCLIP
            control = math.StatisticsControl()
            control.setNumSigmaClip(8.0)
            control.setNumIter(1)

            imageStats = math.makeStatistics(image, statFlags, control)
            sigmaclip = imageStats.getResult(math.STDEVCLIP)[0]
            meanclip = imageStats.getResult(math.MEANCLIP)[0]     

#            min = imageStats.getResult(math.MIN)[0]
#            max = imageStats.getResult(math.MAX)[0]
#            mean = imageStats.getResult(math.MEAN)[0]
#            sigma = imageStats.getResult(math.STDEV)[0]
#            from my_image_tools import FastHistogramImageData_Print
#            hist, d1, d2 = FastHistogramImageData_Print(image.getArray(), '/home/mmmerlin/temp/imagehist.png',meanclip - (3*sigma),meanclip + (3*sigma))
#            hist.Fit('gaus')

            ADU_Values[amp] += meanclip
            Noise_factors[amp] += sigmaclip
    
    ADU_Values /= float(len(bias_files))
    Noise_factors /= float(len(bias_files))
    
    return ADU_Values, Noise_factors
    

