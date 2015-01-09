from time import sleep
from subprocess import Popen
from array import array
from ROOT import *
import ROOT
import math
from matplotlib.pyplot import hist


gROOT.SetBatch(1) #don't show drawing on the screen along the way
gStyle.SetOptStat(111111)
gStyle.SetLineScalePS(0.5) # fixes the line width in vector graphics outputs
#gROOT.LoadMacro("/mnt/hgfs/VMshared/Code/git/my_functions/langaus.C")
#gROOT.LoadMacro("/mnt/hgfs/VMshared/Code/git/my_functions/langaus_plus_gaus.C")

gROOT.LoadMacro("langaus_plus_gaus.C")
gROOT.LoadMacro("langaus.C")



SHOW_FIT_PARS_ON_GRAPH = True
CREATE_IMAGES = True
FILE_TYPE = ".png"
OUTPUT_PATH = "/home/mmmerlin/output/PSF/"
GRAPHS_BLACK_AND_WHITE = False
PAUSE_AT_END = 120 #keep ROOT windows open for X seconds

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1000

#ROOTFILENAME = "F:\Data\Data1.root" #do not put spaces in file name or path
#ROOTfile = TFile.Open(ROOTFILENAME, "RECREATE")


def TruncateTo_n_SigFigs(input, nSigFigs):
    input_char_array = str(input)
    outstring = ""

    if input_char_array[0] == "-": nSigFigs += 1
    
    for i in range(0,len(input_char_array)):
        if input_char_array[i] == ".":
            if i < nSigFigs:
                outstring += "."
                nSigFigs += 1
                continue
            else:
                break

        if i < nSigFigs:
            outstring += input_char_array[i]
        else:
            if i<= math.floor(math.log10(math.fabs(input))) + 1:
                outstring += "0"
    return outstring
def GetFWHM(hist):
    left = right = FWHM = 0.
    maxbin = hist.GetMaximumBin()
    maxbinval = hist.GetBinContent(maxbin)
    halfmax = maxbinval / 2.   

    for i in range (maxbin,0,-1):
        if hist.GetBinContent(i) <= halfmax:
            left = hist.GetBinCenter(i)
            break
    rightmost_bin = hist.GetNbinsX()
    for i in range (maxbin,rightmost_bin):
        if hist.GetBinContent(i) <= halfmax:
            right = hist.GetBinCenter(i)
            break

    FWHM = right - left
    return FWHM, left, right


def GetFirstBinBelowX(hist, x):
    left = 999999
    right = -999999
    
    maxbin = hist.GetMaximumBin()
    maxbinval = hist.GetBinContent(maxbin)
    print "Maxbin num = %s, at value of %s, content = %s" %(maxbin, hist.GetBinCenter(maxbin), maxbinval)
    
    assert maxbinval >= x
    
    for i in range (maxbin,0,-1):
        if hist.GetBinContent(i) <= x:
            left = hist.GetBinCenter(i)
            break
    rightmost_bin = hist.GetNbinsX()
    for i in range (maxbin,rightmost_bin):
        if hist.GetBinContent(i) <= x:
            right = hist.GetBinCenter(i)
            break
        
    # in case the value was never fallen below return histogram edges
    if left == 999999: left = 0
    if right == -999999: right = rightmost_bin
    
    return left, right


def GetLeftRightBinsAtPercentOfMax(hist, percent_level):
    left = 999999.0
    right = -999999.0
    
    maxbin = hist.GetMaximumBin()
    maxbinval = hist.GetBinContent(maxbin)
    print "Maxbin num = %s, at value of %s, content = %s" %(maxbin, hist.GetBinCenter(maxbin), maxbinval)
    
    thr = maxbinval * (float(percent_level)/100.)
    
    for binnum in range (maxbin,0,-1):
        if hist.GetBinContent(binnum) <= thr:
            left = hist.GetBinCenter(binnum)
            break
    rightmost_bin = hist.GetNbinsX()
    for binnum in range (maxbin,rightmost_bin):
        if hist.GetBinContent(binnum) <= thr:
            right = hist.GetBinCenter(binnum)
            break
        
    # in case the value was never fallen below return histogram edges
    if left == 999999.0: left = 0
    if right == -999999.0: right = rightmost_bin
    
    return left, right

def GetLastBinAboveX(hist, x):
    left = 99999999
    right = -99999999
    
    nbins = hist.GetNbinsX()
    
    for i in range (1,nbins):
        if hist.GetBinContent(i) >= x:
            left = hist.GetBinLowEdge(i)
            break
    for i in range (nbins,1, -1):
        if hist.GetBinContent(i) >= x:
            right = hist.GetBinLowEdge(i)+ hist.GetBinWidth(i)
            break
        
    # in case the value was never found return histogram edges
    if left == 99999999: left = hist.GetBinLowEdge(1)
    if right == -99999999: right = hist.GetBinLowEdge(nbins) + hist.GetBinWidth(nbins)
    
    return left, right


def LanGausPlusGausFit(hist, fitmin, fitmax):

    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(fitmin*CHARGE_ADJUSTMENT)
    fitrange.append(fitmax*CHARGE_ADJUSTMENT)

    print ("Integral = " + str(hist.GetSum()))

    initial_pars.append(hist.GetRMS()/7)
    initial_pars.append(hist.GetBinCenter(hist.GetMaximumBin()))
    initial_pars.append(1200000)
    initial_pars.append(29.8*CHARGE_ADJUSTMENT)
    parlimits_lo.append(0)
    parlimits_lo.append(0)
    parlimits_lo.append(0)
    parlimits_lo.append(0)#gaus sigma
    parlimits_hi.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(0)#gaus sigma
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_ChiSqr.append(0)
    ret_NDF.append(0)

    #extra pars
    initial_pars.append(500)#noise gaus height
    initial_pars.append(-1.1*CHARGE_ADJUSTMENT)#noise gaus mean
    initial_pars.append(25.*CHARGE_ADJUSTMENT)#noise gaus sigma
    parlimits_lo.append(0)
    parlimits_lo.append(-2.)
    parlimits_lo.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(2.)
    parlimits_hi.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)

    print ("Initial pars:\n")
    print ("par0 = " + str(initial_pars[0]))
    print ("par1 = " + str(initial_pars[1]))
    print ("par2 = " + str(initial_pars[2]))
    print ("par3 = " + str(initial_pars[3]))
    print ("par4 = " + str(initial_pars[4]))
    print ("par5 = " + str(initial_pars[5]))
    print ("par6 = " + str(initial_pars[6]))


    convolfunc = ROOT.langaus_plus_gaus_fit(hist,fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr, ret_NDF)
    convolfunc.SetNpx(1000)
    convolfunc.SetLineStyle(1)
    convolfunc.SetLineColor(kRed)
    convolfunc.SetLineWidth(LINE_WIDTH)

    try:
        chisqr_over_NDF = ret_ChiSqr[0]/ ret_NDF[0]
        print ("Chisqr = " + str(ret_ChiSqr[0]))
        print ("NDF = " + str(ret_NDF[0]))
        print ("Chisqr / NDF = " + str(chisqr_over_NDF))
    except:
        print "\n\n\n\Some sort of Division by zero error probably occured...\n\n"
 
    print ("Final pars:\n")
    print ("par0 = " + str(ret_pars[0]))
    print ("par1 = " + str(ret_pars[1]))
    print ("par2 = " + str(ret_pars[2]))
    print ("par3 = " + str(ret_pars[3]))
    print ("par4 = " + str(ret_pars[4]))
    print ("par5 = " + str(ret_pars[5]))
    print ("par6 = " + str(ret_pars[6]))


    return convolfunc


def LanGausPlusGausFit_Fixed_MPV(hist, fitmin, fitmax, MPV_Fix):

    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(fitmin)
    fitrange.append(fitmax)

    print ("Integral = " + str(hist.GetSum()))

    initial_pars.append(hist.GetRMS()/7)
    initial_pars.append(MPV_Fix)
    initial_pars.append(1200000)
    initial_pars.append(29.8)
    parlimits_lo.append(0)
    parlimits_lo.append((MPV_Fix)-0.001)
    parlimits_lo.append(0)
    parlimits_lo.append(0)#gaus sigma
    parlimits_hi.append(0)
    parlimits_hi.append((MPV_Fix)+0.001)
    parlimits_hi.append(0)
    parlimits_hi.append(0)#gaus sigma
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_ChiSqr.append(0)
    ret_NDF.append(0)

    #extra pars
    initial_pars.append(500)#noise gaus height
    initial_pars.append(-1.1)#noise gaus mean
    initial_pars.append(25.)#noise gaus sigma
    parlimits_lo.append(0)
    parlimits_lo.append(-2.)
    parlimits_lo.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(2.)
    parlimits_hi.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)

    print ("Initial pars:\n")
    print ("par0 = " + str(initial_pars[0]))
    print ("par1 = " + str(initial_pars[1]))
    print ("par2 = " + str(initial_pars[2]))
    print ("par3 = " + str(initial_pars[3]))
    print ("par4 = " + str(initial_pars[4]))
    print ("par5 = " + str(initial_pars[5]))
    print ("par6 = " + str(initial_pars[6]))


    convolfunc = ROOT.langaus_plus_gaus_fit(hist,fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr, ret_NDF)
    convolfunc.SetNpx(1000)
    convolfunc.SetLineStyle(1)
    convolfunc.SetLineColor(kRed)
    convolfunc.SetLineWidth(LINE_WIDTH)

    try:
        chisqr_over_NDF = ret_ChiSqr[0]/ ret_NDF[0]
        print ("Chisqr = " + str(ret_ChiSqr[0]))
        print ("NDF = " + str(ret_NDF[0]))
        print ("Chisqr / NDF = " + str(chisqr_over_NDF))
    except:
        print "\n\n\n\Some sort of Division by zero error probably occured...\n\n"
    #conv_CCD = ret_pars[1]
    #conv_CCD_error = ret_errors[1]
    #conv_gaus_sigma_in_electrons = ret_pars[3] * 36
    #conv_gaus_sigma_in_electrons_error = ret_errors[3] * 36

    
    #noise_gaus = TF1("noise_gaus","gaus", -100,100)
    #h_sc_ex.Fit("noise_gaus", "0", "",-80,40)
    #noise_gaus.SetNpx(1000)
    #noise_gaus.SetLineStyle(4)
    #noise_gaus.SetLineColor(13) #grey
    #noise_gaus.SetLineWidth(LINE_WIDTH)

    #noise_gaus_pars = noise_gaus.GetParameters()
    #noise_gaus_sigma_in_electrons = noise_gaus_pars[2] *36


    #hist.Draw("")
    #convolfunc.Draw("same")

    #rootlandaufit.Draw("same")
    #noise_gaus.Draw("same")
    #deconvoluted_landau.Draw("same")

    #deconv_landau_CCD_Line.Draw("")
    #rootlandaufit_CCD_Line.Draw("")
    #conv_maxval_line.Draw("")

    print ("Final pars:\n")
    print ("par0 = " + str(ret_pars[0]))
    print ("par1 = " + str(ret_pars[1]))
    print ("par2 = " + str(ret_pars[2]))
    print ("par3 = " + str(ret_pars[3]))
    print ("par4 = " + str(ret_pars[4]))
    print ("par5 = " + str(ret_pars[5]))
    print ("par6 = " + str(ret_pars[6]))


    #linearity_legend = TLegend(0.59,0.70,0.89,0.87) #position the legend at top left
    #linearity_legend.AddEntry(convolfunc,   "Convolution Fit","l")
    #linearity_legend.AddEntry(deconv_landau,"Deconvoluted Landau","l")
    #linearity_legend.AddEntry(rootlandaufit,"ROOT Landau Fit","l")
    #linearity_legend.AddEntry(noise_gaus,   "Gaussian fit to noise","l")
    #linearity_legend.AddEntry("text1",      "Convolution gaussian width = " + str(round(abs(conv_gaus_sigma_in_electrons),1)) + "e^{-}","")
    #linearity_legend.AddEntry("text2",      "Noise gaussian width = " + str(round(abs(noise_gaus_sigma_in_electrons),1)) + "e^{-}","")
    #linearity_legend.Draw()


    return convolfunc



def LandauFit(hist, fitmin, fitmax, no_draw = True):

    fitfunc = TF1("landfunc","landau",fitmin,fitmax)
    fitfunc.SetNpx(10000)
    if no_draw:
        hist.Fit(fitfunc,'0')
    else:
        hist.Fit(fitfunc)
    return fitfunc

def LanGausFit(hist, fitmin, fitmax):
    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(fitmin)
    fitrange.append(fitmax)

    initial_pars.append(hist.GetRMS()/6)
    initial_pars.append(hist.GetBinCenter(hist.GetMaximumBin()))
    initial_pars.append(hist.GetSum())
    initial_pars.append(1.5)
    parlimits_lo.append(0.001)
    parlimits_lo.append(0.001)
    parlimits_lo.append(0.001)
    parlimits_lo.append(0.001)#gaus sigma
    parlimits_hi.append(500)
    parlimits_hi.append(5000)
    parlimits_hi.append(9999999999)
    parlimits_hi.append(1000)#gaus sigma
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_ChiSqr.append(0)
    ret_NDF.append(0)

#    print ("Initial pars:\n")
#    print ("par0 = " + str(initial_pars[0]))
#    print ("par1 = " + str(initial_pars[1]))
#    print ("par2 = " + str(initial_pars[2]))
#    print ("par3 = " + str(initial_pars[3]))

    print 'LanGaus Fit:'
    convolfunc = ROOT.langaufit(hist,fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr, ret_NDF)
    convolfunc.SetNpx(1000)
    convolfunc.SetLineStyle(1)
    convolfunc.SetLineColor(kRed)
    convolfunc.SetLineWidth(LINE_WIDTH)

    try:
        chisqr_over_NDF = ret_ChiSqr[0]/ret_NDF[0]
    except:
        chisqr_over_NDF = -1
    print "Chisqr / NDF = " + str(ret_ChiSqr[0]) + ' / ' + str(ret_NDF[0]) + ' = ' + str(chisqr_over_NDF) +'\n'

#    print ("Final pars:\n")
#    print ("par0 = " + str(ret_pars[0]))
#    print ("par1 = " + str(ret_pars[1]))
#    print ("par2 = " + str(ret_pars[2]))
#    print ("par3 = " + str(ret_pars[3]))

    return convolfunc, chisqr_over_NDF


def GetDeconvolutedLandau(ConvolutedLandauTF1, DeconvolutedLandau):
    pars = ConvolutedLandauTF1.GetParameters()
    
    DeconvolutedLandau.SetParameters(pars[2],pars[1] - (pars[0] * -0.22278298),pars[0])
    #DeconvolutedLandau.SetParameters(pars[2],pars[1],pars[0])
    ScaleLandau(DeconvolutedLandau,ConvolutedLandauTF1)
    
    convol_landau_mpv = ConvolutedLandauTF1.GetParameter(1)
    new_mpv = DeconvolutedLandau.GetMaximumX()

    if (abs(convol_landau_mpv-new_mpv) <= 0.01):
        print ("\nCorrect MPV found")
    else:
        print ("\nERROR in deconvolution matching")

    return DeconvolutedLandau

def MakeLanGausFunc(sigma, mpv, area, gauswidth): #opens object in a TBrowser
    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(-100*CHARGE_ADJUSTMENT)
    fitrange.append(2000*CHARGE_ADJUSTMENT)


    dummy_hist = TH1F("temp", "temp", 200, -100, 1500)
    dummy_hist.Fill(mpv)

    initial_pars.append(sigma)
    initial_pars.append(mpv)
    initial_pars.append(area)
    initial_pars.append(gauswidth)
    parlimits_lo.append(0)
    parlimits_lo.append(0)
    parlimits_lo.append(0)
    parlimits_lo.append(0)#gaus sigma
    parlimits_hi.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(0)
    parlimits_hi.append(0)#gaus sigma
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_pars.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_errors.append(0)
    ret_ChiSqr.append(0)
    ret_NDF.append(0)

    convolfunc = ROOT.langaufit(dummy_hist,fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr, ret_NDF)
    convolfunc.SetNpx(1000)
    convolfunc.SetLineStyle(1)
    convolfunc.SetLineColor(kRed)
    convolfunc.SetLineWidth(LINE_WIDTH)

    convolfunc.SetParameters(sigma,mpv,area,gauswidth)

    return convolfunc





def Browse(object): #opens object in a TBrowser
    from ROOT import TBrowser
    import subprocess
    
#    filename = "temp.root"
##    ROOTfile = TFile.Open(filename, "RECREATE")
#    ROOTfile = TFile.Open('/home/mmmerlin/output/' + filename, "RECREATE")
#    object.Write()
#    ROOTfile.Close()
#    
##    gROOT.SetBatch(0) #don't show drawing on the screen along the way
    
    b = TBrowser()
    b.Add(object)
#    
#    subprocess.call("root " + str(filename))
    
    from time import sleep
    sleep(100)
#    gROOT.SetBatch(1) #don't show drawing on the screen along the way
    



def Finally():
    print("\n\n*******\nEnd of code\n*******")
    sleep(PAUSE_AT_END)
    

#def sig2g(x, par):
#    xx    = x[0]
#    norm  = par[0]                      // par(0) - norm
#    mean1 = par[1]                      // par(1) - mean
#    sigm1 = par[2]                      // par(2) - sigm
#    r     = par[3]                      // par(7) - norm
#    mean2 = par[4]                      // par(8) - mean
#    sigm2 = par[5]                      // par(9) - sigm
#    
#    x12 = (xx - mean1) * (xx - mean1)
#    sig12 = sigm1 * sigm1
#    sig1 = 1.  / (sigm1 * sqrt(2. * math.pi)) * exp(-x12 / (2. * sig12))
#
#    // 2nd gaus
#    x22 = (xx - mean2) * (xx - mean2)
#    sig22 = sigm2 * sigm2
#    sig2 = abs(r) / (sigm2 * sqrt(2. * math.pi)) * exp(-x22 / (2. * sig22))
#    //Double_t sig2 = (1-r) / (sigm2 * sqrt(2. * math.pi)) * exp(-x22 / (2. * sig22))
#    
#    out = (sig1 + sig2) * norm /(1.+ abs(r))
#    return out



# Just a Gaussian
#def double_gaus(y, q):
#    result = q[2]/sqrt(2.*math.pi*q[1]*q[1])*exp(-0.5*(y[0]-q[0])*(y[0]-q[0])/(q[1]*q[1])) + q[5]/sqrt(2.*math.pi*q[4]*q[4])*exp(-0.5*(y[3]-q[3])*(y[3]-q[3])/(q[4]*q[4]))
#    return result 
  

def DoubleGausFit(hist, fitmin, fitmax):
    fitfunc =  TF1("double_gaus","([2]/TMath::Sqrt(2.*3.14159265359*[1]*[1])*TMath::Exp(-0.5*(x-[0])*(x-[0])/([1]*[1]))) + ([5]/TMath::Sqrt(2.*3.14159265359*[4]*[4])*TMath::Exp(-0.5*(x-[3])*(x-[3])/([4]*[4])))", fitmin,fitmax)
    fitfunc.SetNpx(1000)
    
    integral = hist.GetSum()

    fitfunc.SetParameter(0,hist.GetBinCenter(hist.GetMaximumBin())) # mean
    fitfunc.SetParameter(1,8) # sigma
    fitfunc.SetParameter(2,integral*2.) # height
    fitfunc.SetParameter(3,hist.GetBinCenter(hist.GetMaximumBin())+50) # mean
    fitfunc.SetParameter(4,8) # sigma
    fitfunc.SetParameter(5,integral/4) # height

    fitfunc.SetParLimits(0,hist.GetBinCenter(hist.GetMaximumBin())-50,hist.GetBinCenter(hist.GetMaximumBin())+50 ) # mean
    fitfunc.SetParLimits(1,1,50) # sigma
    fitfunc.SetParLimits(2,integral/100,integral*5) # height
    fitfunc.SetParLimits(3,hist.GetBinCenter(hist.GetMaximumBin())+10,hist.GetBinCenter(hist.GetMaximumBin())+200) # mean
    fitfunc.SetParLimits(4,1,50) # sigma
    fitfunc.SetParLimits(5,integral/100,integral*5) # height


    # for REB readout gains
#     fitfunc.SetParameter(0,hist.GetBinCenter(hist.GetMaximumBin())) # mean
#     fitfunc.SetParameter(1,50) # sigma
#     fitfunc.SetParameter(2,integral*2.) # height
#     fitfunc.SetParameter(3,hist.GetBinCenter(hist.GetMaximumBin())+250) # mean
#     fitfunc.SetParameter(4,50) # sigma
#     fitfunc.SetParameter(5,integral/4) # height
#     
#     fitfunc.SetParLimits(0,hist.GetBinCenter(hist.GetMaximumBin())-100,hist.GetBinCenter(hist.GetMaximumBin())+100 ) # mean
#     fitfunc.SetParLimits(1,1,100) # sigma
#     fitfunc.SetParLimits(2,integral/100,integral*10) # height
#     fitfunc.SetParLimits(3,hist.GetBinCenter(hist.GetMaximumBin())-400,hist.GetBinCenter(hist.GetMaximumBin())+1000) # mean
#     fitfunc.SetParLimits(4,1,100) # sigma
#     fitfunc.SetParLimits(5,integral/100,integral*5) # height

    hist.Fit(fitfunc,"ME0","", fitmin, fitmax)
    
    chisq = fitfunc.GetChisquare()
    NDF = fitfunc.GetNDF()
    
    try:
        chisqr_over_NDF = chisq/NDF
    except:
        chisqr_over_NDF = -1
    print "Chisqr / NDF = " + str(chisq) + ' / ' + str(NDF) + ' = ' + str(chisqr_over_NDF) +'\n'

    return fitfunc, chisqr_over_NDF




def ListToHist(list, savefile, log_z = False, nbins = 20, histmin = None, histmax = None):
    from ROOT import TCanvas, TH1F
    import numpy as np
    c1 = TCanvas( 'canvas', 'canvas', 500, 200, 700, 500 ) #create canvas
    if histmin == None: histmin = min(list)
    if histmax == None: histmax = max(list)
    
    hist = TH1F('', '',nbins,histmin,histmax)
    
    for value in list:
        if (value == np.inf) or (value == np.nan):
            print 'excluded inf/nan value'
            continue
        hist.Fill(value)
    hist.Draw()

    if log_z: c1.SetLogz()
#        image_hist.SetStats(False)
    c1.SaveAs(savefile)
    return 

def ListVsList(list_x, list_y, savefile, xmin = None, xmax = None, xtitle = '', ytitle = '', setlogy = False):
    from ROOT import TCanvas, TGraph
    import numpy
    
    c1 = TCanvas( 'canvas', 'canvas', 500, 200, 700, 500 ) #create canvas
    
    if len(list_x) != len(list_y):
        print "ERROR - x and y sets are different sizes"
        print str(savefile) + " was therefore not created"
        return
    
    if xmin == None: xmin = min(list_x)
    if xmax == None: xmax = max(list_x)
        
    graph = TGraph(len(list_x), numpy.asarray(list_x, dtype = 'f8'), numpy.asarray(list_y, dtype = 'f8')) #populate graph with data points
    graph.SetTitle('')
    graph.GetXaxis().SetRangeUser(xmin,xmax)
    graph.GetXaxis().SetTitle(xtitle)
    graph.GetYaxis().SetTitle(ytitle)
    graph.SetMarkerColor(2)
    graph.SetMarkerStyle(2)
    graph.SetMarkerSize(1.5)
    graph.Draw("AP")
    if setlogy: c1.SetLogy()
    c1.SaveAs(savefile)
    return 






if __name__=="__main__": # main code block
    

    print "You are running the wrong file!"
    print "End of python code"
    Finally()
