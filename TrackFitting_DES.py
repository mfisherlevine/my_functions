import lsst.afw.geom.ellipses as Ellipses
import lsst.afw.detection   as afwDetect
import numpy as np
from lsst import afw
from __builtin__ import str
from numpy import math, float64
from IPython.parallel.controller.scheduler import numpy

midline = 1000
edge_left = 0
edge_right = 2000 #what is up with this giving such low numbers but not zero?!
edge_bottom = 0 #why does 0 here give no results, but not the same with left?
edge_top = 4106

edge_track_stamp_border = 0

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1000

DEBUG = False
from root_functions import OUTPUT_PATH, FILE_TYPE, GetLastBinAboveX
from ROOT import TH1F, TH2F, TF1, TCanvas

class TrackStats:
    """Properties of a cosmic ray candidate"""
    def __init__(self):
        self.data = None
        self.filename = None
        
        self.ellipse_a = None
        self.ellipse_b = None
        self.ellipse_theta = None

        self.ellipse_Ixx = None
        self.ellipse_Iyy = None
        self.ellipse_Ixy = None
        
        self.centroid_x = None
        self.centroid_y = None
        self.BBox = None
        self.left = None
        self.right = None
        self.bottom = None
        self.top = None
        self.xsize = None
        self.ysize = None
        
        self.left_track = False
        self.right_track = False
        self.bottom_track = False
        self.top_track = False
        self.midline_track = False
        
        self.diagonal_length_pixels = None
        self.length_x_um = None
        self.length_y_um = None
        self.length_true_um = None
        self.track_angle_to_vertical_degrees = None

        self.LineOfBestFit = StraightLine()

        self.flux = None
        self.npix = None
        self.de_dx = None
        self.discriminator = None
        
#        self.pixel_list_all_in_footprint = None
#        self.pixel_list_all_in_bbox = None
        
#        self.is_cosmic = False
#        self.is_worm = False
#        self.is_spot = False
    
class StraightLine:
    """Defines a straight line where y = ax + b and also gives its correlation coefficient"""
    def __init__(self, a = None, b = None, a_error = None, b_error = None, R2 = None, chisq = None, NDF = None, chisq_red = None):
        self.a = a
        self.a_error = a
        self.b = b
        self.b_error = b
        self.R2 = R2
        self.chisq = chisq
        self.NDF = NDF
        self.chisq_red = chisq_red

def _GetFlux(footprint):
#    import numpy as np
#    array = np.zeros(footprint.getArea(), dtype=bgSubImg.getArray().dtype)#MERLIN1
#    afwDetect.flattenArray(footprint, bgSubImg.getArray(), array, bgSubImg.getXY0())#MERLIN1
#    flux = array.sum() #MERLIN1
    
    data = footprint.getImageArray()
    return data.sum(dtype = 'f8')

def GetEdgeType(stat):
        if IsEdgeTrack(stat) == False: return "none"
        if stat.right_track == True: return "right"
        elif stat.left_track == True: return "left"
        elif stat.top_track == True: return "top"
        elif stat.bottom_track == True: return "bottom"
        elif stat.midline_track == True: return "midline"
        else: return "error"

def IsEdgeTrack(stat):
        if stat.right_track == True:
            return True
        elif stat.left_track == True:
            return True
        elif stat.top_track == True:
            return True
        elif stat.bottom_track == True:
            return True
        elif stat.midline_track == True:
            return True
        else: return False
        
def FitStraightLine(data, ncols_exclude = 0):
############################################# graph method
#    xpoints, ypoints = array('d'), array('d')
#    for x in range(data.shape[0]):
#        for y in range(data.shape[1]):
#            value = data[x,y]
#            if value != 0:
#                xpoints.append(float(x))
#                ypoints.append(float(y))
#
#    c3 = TCanvas( 'canvas', 'canvas', 500, 200, 700, 500 ) #create canvas
#    image_graph = TGraph(len(xpoints), xpoints, ypoints)
#    fit_func = TF1("line","pol1", -100, 100)
#    image_graph.Fit(fit_func, "", "")
#    image_graph.Draw("AP")
#    fit_func.Draw("same")
#    c3.SaveAs(OUTPUT_PATH + "image_graph" + FILE_TYPE)
#    del xpoints,ypoints
    
    
############################################# 2D hist method
#    nbinsx = xmax = data.shape[0]
#    nbinsy = ymax = data.shape[1]

    data = data[ncols_exclude:,]

    nbinsx = xmax = max(data.shape[0], data.shape[1])
    nbinsy = ymax = max(data.shape[0], data.shape[1])
    xlow = 0
    ylow = 0
    image_hist = TH2F('hist', 'hist',nbinsx,xlow,xmax,nbinsy, ylow, ymax)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                image_hist.Fill(float(x),float(y),float(value))
#                image_hist.Fill(float(x),float(y),1) #for not weighting by intensity

#    fit_func = TF1("line","pol1", -100, 100) # suspect this is quicker than using a custom TF1 WARNING - THIS FAILS A LOT OF THE TIME FOR SOME REASON - DO NOT USE
    fit_func = TF1("line","[1]*(x+"+str(ncols_exclude)+") + [0]", xlow - 2, xmax + 2)# NB If you declare a TF1 parameters should be [1]*x + [0] in order to match the return pars from "pol1"
    fit_func.SetNpx(1000)
    image_hist.Fit(fit_func, "MEQ0", "")
    
    a = fit_func.GetParameter(1) 
    a_error = fit_func.GetParError(1)
    b = fit_func.GetParameter(0)
    b_error = fit_func.GetParError(0)
    R2 = image_hist.GetCorrelationFactor()**2
    
    chisq = fit_func.GetChisquare()
    NDF = fit_func.GetNDF()
    try:
        chisqr_over_NDF = chisq/NDF
    except:
        chisqr_over_NDF = -1  
    
    line_of_fit = StraightLine(a, b, a_error, b_error, R2, chisq, NDF, chisqr_over_NDF)

    if DEBUG:
        c4 = TCanvas( 'canvas', 'canvas', CANVAS_WIDTH,CANVAS_HEIGHT)
#        image_hist.GetXaxis().SetRangeUser(0,data.shape[0])
#        image_hist.GetYaxis().SetRangeUser(0,data.shape[1])
        image_hist.Draw("lego2") #box, lego, colz, lego2 0
        fit_func.Draw("same")
#        c4.SetLogz()
        image_hist.SetStats(False)
        c4.SaveAs(OUTPUT_PATH + "image_graph_hist_" + str(DEBUG_TRACK_NUM) + FILE_TYPE)
        print "grad = ", str(a)
        print "grad error= ", str(a_error)
        print "inercept = ", str(b)
        print "inercept error = ", str(b_error)
        print "R^2 = %s"%R2

    return line_of_fit

def MeasurePSF_Whole_track(data, fitted_line):
    #get new coefficients as distance to line uses straight line of form ax + by + c = 0
    a = -1. * fitted_line.a
    b = 1
    c = -1. * fitted_line.b
    
    histmin = -1 * max(data.shape[0], data.shape[1])
    histmax = max(data.shape[0], data.shape[1])
    nbins = (histmax - histmin) * 2
    
    psf_hist = TH1F('Whole Track', 'Whole Track', nbins, histmin, histmax)
    
    
    for xcoord in range(data.shape[0]):
        for ycoord in range(data.shape[1]):
            x = xcoord + 0.5 #adjust for bin centers - NB This is important!
            y = ycoord + 0.5 #adjust for bin centers - NB This is important!
            value = float(data[xcoord,ycoord])
#            if value > 0:
#            distance = abs(a*x + b*y + c) / (a**2 + b**2)**0.5
            non_abs_distance = (a*x + b*y + c) / (a**2 + b**2)**0.5
            psf_hist.Fill(non_abs_distance, value)
            
#    for i in range(nbins):
#        psf_hist.SetBinError(i,1)
            
    fitmin, fitmax = GetLastBinAboveX(psf_hist, 0.1)
    viewmin, viewmax = GetLastBinAboveX(psf_hist, 0.1)
    psf_hist.GetXaxis().SetRangeUser(viewmin,viewmax)

    fit_func = TF1("gaus", "gaus", fitmin, fitmax)
    fit_func.SetNpx(1000)
    psf_hist.Fit(fit_func, "WME0Q", "", fitmin, fitmax)
    
    sigma = fit_func.GetParameter(1) 
    sigma_error = fit_func.GetParError(1)
    
    chisq = fit_func.GetChisquare()
    NDF = fit_func.GetNDF()
    try:
        chisqr_over_NDF = chisq/NDF
    except:
        chisqr_over_NDF = -1  
        
    
    if DEBUG:
        print "PSF gaus Chi2 = \t%s"%chisqr_over_NDF
        c1 = TCanvas( 'canvas', 'canvas', CANVAS_WIDTH,CANVAS_HEIGHT)
        psf_hist.Draw("")
        fit_func.Draw("same")
        try:
            c1.SaveAs(OUTPUT_PATH + "psf_hist_" + str(DEBUG_TRACK_NUM) + FILE_TYPE)
        except:
            c1.SaveAs(OUTPUT_PATH + "psf_hist_" + FILE_TYPE)
            
    return








def GetSecNum(data, xcoord, ycoord, nsecs, FitLine):
    secnum = -1
    xmin = 0.
    xmax = data.shape[0]
    
    xfraction = xcoord / xmax
    secsize = 1. / nsecs
    
    secnum = int(math.floor(xfraction / secsize))    
    
    return secnum
    



def MeasurePSF_in_Sections(data, fitted_line, nsecs = 3, tgraph_filename = ''):
    #get new coefficients as distance to line uses straight line of form ax + by + c = 0
    a = -1. * fitted_line.a
    b = 1
    c = -1. * fitted_line.b
    
    histmin = -1 * max(data.shape[0], data.shape[1])
    histmax = max(data.shape[0], data.shape[1])
    nbins = (histmax - histmin) * 2
    
    hists = []
    for i in range(nsecs):
       hists.append(TH1F('Track section ' + str(i), 'Track section ' + str(i), nbins, histmin, histmax))
    
    
    for xcoord in range(data.shape[0]):
        for ycoord in range(data.shape[1]):
            x = xcoord + 0.5 #adjust for bin centers - NB This is important!
            y = ycoord + 0.5 #adjust for bin centers - NB This is important!
            secnum = GetSecNum(data, x, y, nsecs, None) # TODO!
            
            value = float(data[xcoord,ycoord])
#            if value < 800:
            non_abs_distance = (a*x + b*y + c) / (a**2 + b**2)**0.5
            hists[secnum].Fill(non_abs_distance, value)
        
    sigmas, sigma_errors = [], []
            
    for i, hist in enumerate(hists):
        fitmin, fitmax = GetLastBinAboveX(hist, 0.1)
#        viewmin, viewmax = GetLastBinAboveX(hist, 0.1)
        viewmin, viewmax = -2,2
        hist.GetXaxis().SetRangeUser(viewmin,viewmax)
    
        fit_func = TF1("gaus", "gaus", fitmin, fitmax)
        fit_func.SetNpx(1000)
        hist.Fit(fit_func, "MEQ", "", fitmin, fitmax)
        
        legend_text = []
        
        sigma = fit_func.GetParameter(2) 
        sigma_error = fit_func.GetParError(2)
        
        sigmas.append(abs(15*sigma)) #15 for the 15um per pixel
        sigma_errors.append(abs(15*sigma_error)) #15 for the 15um per pixel
        
        mean = fit_func.GetParameter(15*1) #15 for the 15um per pixel
        mean_error = fit_func.GetParError(15*1) #10 for the 15um per pixel
        
        chisq = fit_func.GetChisquare()
        NDF = fit_func.GetNDF()
        try:
            chisqr_over_NDF = chisq/NDF
        except:
            chisqr_over_NDF = -1  
#        if chisqr_over_NDF > 500 or chisqr_over_NDF <= 1:
#            return [], [], []
        
        legend_text.append('mean = ' + str(mean) + ' #pm ' + str(mean_error) + " #mum")
        legend_text.append('#sigma = ' + str(round(sigma,4)) + ' #pm ' + str(round(sigma_error,4)) + " #mum")
    
        if DEBUG: #For showing each of n PSF *Histograms* per track
            c1 = TCanvas( 'canvas', 'canvas', CANVAS_WIDTH,CANVAS_HEIGHT)
            hist.Draw("")
            if legend_text != '':
                from ROOT import TPaveText
                textbox = TPaveText(0.0,1.0,0.2,0.8,"NDC")
                for line in legend_text:
                    textbox.AddText(line)
                textbox.SetFillColor(0)
                textbox.Draw("same")
            c1.SaveAs(OUTPUT_PATH + "psf_section_" + str(i) + FILE_TYPE)


    from ROOT import TGraphErrors
    c2 = TCanvas( 'canvas', 'canvas', CANVAS_WIDTH,CANVAS_HEIGHT)
    assert nsecs == len(sigmas) == len(sigma_errors)
    xpoints = GenXPoints(nsecs, 250.)
    
    gr = TGraphErrors(nsecs, np.asarray(xpoints,dtype = float), np.asarray(sigmas,dtype = float), np.asarray([0 for i in range(nsecs)],dtype = float), np.asarray(sigma_errors,dtype = float)) #populate graph with data points
    gr.SetLineColor(2)
    gr.SetMarkerColor(2)
    gr.Draw("AP")
    
    fit_func = TF1("line","[1]*x + [0]", -1, nsecs+1)
    fit_func.SetNpx(1000)
    gr.Fit(fit_func, "MEQ", "")
    
    a = fit_func.GetParameter(1) 
    a_error = fit_func.GetParError(1)
    
    if DEBUG:
        if tgraph_filename == '': tgraph_filename = OUTPUT_PATH + 'psf_graph' + '.png'
        gr.SetTitle("")
        gr.GetYaxis().SetTitle('PSF #sigma (#mum)')
        gr.GetXaxis().SetTitle('Av. Si Depth (#mum)')
        c2.SaveAs(tgraph_filename)
    
#    if a_error >= a:
#        print "Inconclusive muon directionality - skipped track %s"%tgraph_filename
#        return [],[],[]
   
    for j in range(nsecs):
        if sigma_errors[j] > sigmas[j]:
            print "bad fit skipped"
            return [],[],[]
    
    if a < 0:
        sigmas.reverse()
        sigma_errors.reverse()
        return xpoints, sigmas, sigma_errors
    else:
        return xpoints, sigmas, sigma_errors
    
    
#    return xpoints, sigmas, sigma_errors
    

def GenXPoints(nsecs, thickness):
    xpoints = [thickness*(2*i + 1)/(2*float(nsecs)) for i in range(nsecs)] #not the most intuitive way of expressing that, but it works out the same as the "natural" way. Has been double checked.
    return xpoints

def CalcDeltaParameter(data, fitted_line):
    #get new coefficients as distance to line uses straight line of form ax + by + c = 0
    a = -1. * fitted_line.a
    b = 1
    c = -1. * fitted_line.b
    
    histmin = -1 * max(data.shape[0], data.shape[1])
    histmax = max(data.shape[0], data.shape[1])
    nbins = (histmax - histmin) * 2
    
    discriminator = 0
    for xcoord in range(data.shape[0]):
        for ycoord in range(data.shape[1]):
            x = xcoord + 0.5 #adjust for bin centers - NB This is important!
            y = ycoord + 0.5 #adjust for bin centers - NB This is important!
            value = float(data[xcoord,ycoord])
            distance = abs(a*x + b*y + c) / (a**2 + b**2)**0.5
            try:
                discriminator += value * (math.e**(distance))
            except:
                discriminator = 1e324
                return discriminator
    discriminator /= ((data.shape[0])**2 + (data.shape[1])**2)**0.5        
    
    return discriminator
   
   
def _GetPixelData(footprint, parent_image, technique = 'footprint',):
    if technique != 'footprint' and technique != 'bbox': technique = 'footprint'
    
    if technique == 'bbox':
        print "WARNING - data here may be transposed - check vs other method before relying on x-y orientation!"
        box = footprint.getBBox()
        xmin = box.getMinX()
        xmax = box.getMaxX() + 1
        ymin = box.getMinY()
        ymax = box.getMaxY() + 1
        data = parent_image.getArray()[ymin:ymax,xmin:xmax]
    else:
        import lsst.afw.image as afwImg
        maskedImg = afwImg.MaskedImageF(footprint.getBBox())
        footprint.insert(maskedImg) #this pulls the above-threshold values out and copies them into a numpy array
        data = maskedImg.getImage().getArray().transpose() # transpose so that it has the same orientation as ds9 when indexed with x,y
    return data
        
        


def GetTrackStats(footprint, parent_image, filename, save_track_data = False, track_number = 0):
    global DEBUG_TRACK_NUM 
    DEBUG_TRACK_NUM = track_number
    
    stats = TrackStats() #init stats object
    
    #basic stats properties
    stats.filename = filename
    stats.data = _GetPixelData(footprint, parent_image, 'footprint') #gets cleared later if not being saved

    #ellipse shape stuff   
    quadshape = footprint.getShape()
    stats.ellipse_Ixx = quadshape.getIxx()
    stats.ellipse_Iyy = quadshape.getIyy()
    stats.ellipse_Ixy = quadshape.getIxy()
    axesshape = Ellipses.Axes(quadshape)
    stats.ellipse_a = axesshape.getA()
    stats.ellipse_b = axesshape.getB()
    stats.ellipse_theta = axesshape.getTheta()
    
    #location
    centroid_x, centroid_y = footprint.getCentroid()
    stats.centroid_x = centroid_x
    stats.centroid_y = centroid_y
    stats.BBox = footprint.getBBox()
    
    # edge/midline crossing determination
    stats.left = stats.BBox.getBeginX()
    stats.right = stats.BBox.getEndX()
    stats.bottom = stats.BBox.getBeginY()
    stats.top = stats.BBox.getEndY()
#     if stats.left == edge_left: stats.left_track = True
#     if stats.right == edge_right: stats.right_track = True
#     if stats.top == edge_top: stats.top_track = True
#     if stats.bottom == edge_bottom: stats.bottom_track = True
#     if stats.bottom < midline and stats.top > midline: stats.midline_track = True
    if stats.left <= edge_left: stats.left_track = True
    if stats.right >= edge_right: stats.right_track = True
    if stats.top >= edge_top: stats.top_track = True
    if stats.bottom <= edge_bottom: stats.bottom_track = True
    if (stats.bottom <= midline) and (stats.top >= midline): stats.midline_track = True

    #flux and area
    stats.npix = footprint.getNpix()
    stats.flux = _GetFlux(footprint)
    stats.xsize = footprint.getBBox().getWidth()
    stats.ysize = footprint.getBBox().getHeight()
    stats.BBox_area = stats.xsize * stats.ysize
    assert stats.xsize == stats.data.shape[0] #check for idiocy
    assert stats.ysize == stats.data.shape[1]
    assert stats.right - stats.left == stats.xsize
    assert stats.top - stats.bottom == stats.ysize
    

    #length and deprojection
    stats.length_x_um = footprint.getBBox().getWidth() * 10
    stats.length_y_um = footprint.getBBox().getHeight() * 10
    stats.diagonal_length_pixels = (footprint.getBBox().getWidth()**2 + footprint.getBBox().getHeight()**2)**.5 # diagonal of the bbox - can be refined
    stats.length_true_um = (stats.length_x_um**2 + stats.length_y_um**2 + 100**2)**0.5 # diagonal of bbox - can be refined by replacing refined diagonal with whatever the above refinement is
    stats.de_dx = stats.flux / stats.length_true_um
#    stats.track_angle_to_vertical_degrees = ???
    
  
    #track fitting
    stats.LineOfBestFit = FitStraightLine(stats.data)
 
    #PSF Measurement
#    MeasurePSF_Whole_track(stats.data, stats.LineOfBestFit)
    
    #Deltas
    stats.discriminator = CalcDeltaParameter(stats.data, stats.LineOfBestFit)
       
          
#    import TrackViewer as TV
#    filepath = "/home/mmmerlin/output/PSF/" + str(DEBUG_TRACK_NUM) + ".png"
#    TV.TrackToFile_ROOT(stats.data, filepath, False, 'lego2 0', force_aspect=True)
    
    if not save_track_data: stats.data = None
    
    return stats













#    if save_track_data:
#        stats.pixel_list_all_in_bbox = [] # put all pixels from bbox in list    
#        xmin = stats.BBox.getMinX()
#        xmax = stats.BBox.getMaxX() + 1
#        ymin = stats.BBox.getMinY()
#        ymax = stats.BBox.getMaxY() + 1
#        bboxdata = parent_image.getArray()[ymin:ymax,xmin:xmax]
#        list_of_lists = bboxdata.tolist()
#        for list in list_of_lists:
#            for value in list:
#                stats.pixel_list_all_in_bbox.append(value)
#        
#      
#        stats.pixel_list_all_in_footprint = [] # put all pixel in footprint in list
#        import lsst.afw.image       as afwImg
#        maskedImg = afwImg.MaskedImageF(footprint.getBBox())
#        footprint.insert(maskedImg)
#        footprintdata = maskedImg.getImage().getArray()
#        list_of_lists = footprintdata.tolist()
#        for list in list_of_lists:
#            for value in list:
#                if value <> 0: stats.pixel_list_all_in_footprint.append(value)
        
    



#    print data.shape
#    print "%r"%data
#    import pylab as pl
#    pl.imshow(data, cmap=pl.cm.gray, interpolation='none')
#    pl.show()


