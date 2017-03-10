from time import sleep
from subprocess import Popen
#from ROOT import TCanvas, TF1, TH1F, TFile, TGraph
from array import array
from ROOT import *
import ROOT
import math


gROOT.SetBatch(1) #don't show drawing on the screen along the way
gStyle.SetOptStat(111111)
gStyle.SetLineScalePS(0.5) # fixes the line width in vector graphics outputs
gROOT.LoadMacro("/home/mmmerlin/my_functions/langaus.C")
gROOT.LoadMacro("/home/mmmerlin/my_functions/langaus_plus_gaus.C")

SHOW_FIT_PARS_ON_GRAPH = True
CREATE_IMAGES = True
FILE_TYPE = ".pdf"
#OUTPUT_PATH = "F:\\Data\\output_new\\"
OUTPUT_PATH = "/home/mmmerlin/output/"
GRAPHS_BLACK_AND_WHITE = False
PAUSE_AT_END = 120 #keep ROOT windows open for X seconds
ConversionFactor = (2.2e-12 /1000) / 1.6e-19 # NB the extra 1000 is because the pulse amplitude is in mV not V
CCD_Gain_factor = 1.39342
CHARGE_ON_X_AXIS = True
CHARGE_ADJUSTMENT = (36.0/1000.)

#ROOTFILENAME = "F:\Data\Data1.root" #do not put spaces in file name or path
#ROOTfile = TFile.Open(ROOTFILENAME, "RECREATE")
#profiledir = ROOTfile.mkdir("Amp Profiling")

#def EvalWithError(func):
#    r = TFitResultPtr()
#    r = func. graph->Fit(myFunction,"S");

#double x[1] = { x0 };
#double err[1];  // error on the function at point x0

#r->GetConfidenceIntervals(1, 1, 1, x, err, 0.683, false);
#cout << " function value at " << x[0] << " = " << myFunction->Eval(x[0]) << " +/- " << err[0] << endl;



#    return value, error

def SetGraphStyleScopePlot(graph, maintitle):
    if (GRAPHS_BLACK_AND_WHITE):
        graph.SetLineColor( 1 )
        graph.SetLineWidth( 1 )
    else:
        graph.SetLineColor( 2 )
        graph.SetLineWidth( 1 )
    graph.SetTitle( maintitle )
def SetCanvasStyle(canvas):
    canvas.SetGrid()
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
def PlotScopeTrace(Datafile, rootTitle, Title, xScale = 1, xOffset = 0, delim = ",", no_draw = False, yunit_override = "",  xunit_override = "", opt_draw_option = ""):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    linecontents = []
    linecounter = 0
    ymax = 0.

    for line in localfile.readlines()[6:]: #read in data - skip first 5 lines (for header)
        linecontents = line.split(delim)
        if (line == "\n"): continue
        if len(linecontents) == 5: #deals with different scope formattings and data files where no time scale is given
            x = (float(linecontents[3]) + xOffset) * xScale
            y = linecontents[4]
        elif len(linecontents) == 2:
            x = (float(linecontents[0]) + xOffset) * xScale
            y = linecontents[1]
        elif len(linecontents) == 1:
            x = (float(linecounter) + xOffset) * xScale
            linecounter +=1
            y = linecontents[0]

        xlist.append(float(x))
        ylist.append(float(y))
        if float(y) > ymax: ymax = float(y) # find max to get the y scale

    xunits, yunits = "",""
    if xlist[-1] < 1e-6: #work out units (probably not very robust method)
        xunits = "ns"
    elif xlist[-1] < 1e-3:
        xunits = "#mus" # latex command for printing a greek mu
    else: xunits = "ERROR"
    
    if ymax >= 1:
        yunits = "V"
    elif ymax < 1:
        yunits = "mV"
    else: yunits = "ERROR"

    if not (yunit_override == ""): yunits = yunit_override
     
    for i in range (len(xlist)): # multiply up data points so units make sense
        if xunits == "ns":
            xlist[i] *= 1e9
        elif xunits == "#mus": # latex command for printing a greek mu
            xlist[i] *= 1e6
    for i in range (len(ylist)):
        if yunits == "mV":
            ylist[i] *= 1000
        elif yunits == "V":
            pass
       
    gr = TGraph(len(xlist), xlist, ylist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.GetXaxis().SetRangeUser(xlist[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange

    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)
    if not (no_draw):
        if (opt_draw_option == ""):
            gr.Draw("AC")
        else:
            gr.Draw(opt_draw_option)

    return gr


def PlotSpicefile(Datafile, rootTitle, Title, xScale = 1, xOffset = 0, delim = ",", no_draw = False, yunit_override = "",  xunit_override = "", yScale = 1, plot_differential = False, skiplines = 1):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    linecontents = []
    linecounter = 0
    ymax = 0.

    for line in localfile.readlines()[skiplines:]: #1 header line in LTSpice files
        linecontents = line.split(delim)
        if (line == "\n"): continue
        if len(linecontents) == 2:
            x = (float(linecontents[0]) + xOffset) * xScale
            y = float(linecontents[1]) * yScale


        xlist.append(float(x))
        ylist.append(float(y))
        if float(y) > ymax: ymax = float(y) # find max to get the y scale


    if plot_differential:
        oldlist = array('d')
        oldlist = ylist
        for i in range(0, len(xlist)-1):
            #ylist[i] = (oldlist[i+5] + oldlist[i+4] + oldlist[i+3]) - (oldlist[i+2] + oldlist[i+1] + oldlist[i])
            ylist[i] = (oldlist[i+1] - oldlist[i]) / (xlist[i+1] - xlist[i])
            #ylist[i] = (oldlist[i+1] - oldlist[i])
            #ylist[i] = 1+oldlist[i+1] 

    xunits, yunits = "",""
    if xlist[-1] < 1e-6: #work out units (probably not very robust method)
        xunits = "ns"
    elif xlist[-1] < 1e-3:
        xunits = "#mus" # latex command for printing a greek mu
    else: xunits = "ERROR"
    
    if ymax >= 1:
        yunits = "V"
    elif ymax < 1:
        yunits = "mV"
    else: yunits = "ERROR"

    if not (yunit_override == ""): yunits = yunit_override
     
       
    gr = TGraph(len(xlist), xlist, ylist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.GetXaxis().SetRangeUser(xlist[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange


    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)

    gr.SetLineWidth(4)


    if not (no_draw):
        gr.Draw("AC")
       
    return gr


def PlotGraph(Datafile, rootTitle, Title, xseries_col_num, yseries_col_num, xunits, yunits, skiplines = 0, noDraw = False):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    linecontents = []
    linecounter = 0
    ymax = 0.

    for line in localfile.readlines()[skiplines:]: #read in data - skip first 5 lines (for header)
        linecontents = line.split("\t")
        x = linecontents[xseries_col_num-1]
        y = linecontents[yseries_col_num-1]
        linecounter +=1

        xlist.append(float(x))
        ylist.append(float(y))
        if float(y) > ymax: ymax = float(y) # find max to get the y scale

    print "Number of points =",len(xlist)
    gr = TGraph(len(xlist), xlist, ylist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.SetMarkerStyle(21)
    #gr.GetXaxis().SetRangeUser(xlist.[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange

    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)
    if noDraw == False:
        gr.Draw("AP")
       
    return gr


def PlotGraphDiamondMIP(Datafile, rootTitle, Title, xseries_col_num, yseries_col_num, xunits, yunits, skiplines = 0, noDraw = False):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    linecontents = []
    linecounter = 0
    ymax = 0.

    for line in localfile.readlines()[skiplines:]: #read in data - skip first 5 lines (for header)
        linecontents = line.split("\t")
        x = linecontents[xseries_col_num-1]
        y = linecontents[yseries_col_num-1]
        linecounter +=1

        xlist.append(float(x))
        ylist.append(float(y))
        if float(y) > ymax: ymax = float(y) # find max to get the y scale

    print "Number of points =",len(xlist)
    gr = TGraph(len(xlist), xlist, ylist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.SetMarkerStyle(21)
    #gr.GetXaxis().SetRangeUser(xlist.[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange

    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)
    if noDraw == False:
        gr.Draw("AC")
       
    return gr



def Get_dV_Dt_spice(graph):
    func_crrc =  TF1("func_crrc","[0] + [1]/((TMath::Exp([2]* (x-[3])) + TMath::Exp([4]* (x-[3]))))", 10,200)
    func_crrc.SetNpx(1000)
    #func_crrc.SetLineWidth(10) 
    func_crrc.SetLineColor(kBlack)
    func_crrc.SetLineStyle(2)

    func_crrc.SetParameter(0,-12)
    func_crrc.SetParameter(1,2.1)
    func_crrc.SetParameter(2,.55)
    func_crrc.SetParameter(3,36.9)
    func_crrc.SetParameter(4,.16)

    graph.Fit(func_crrc,"0","")
    func_crrc.SetParameter(0,0)

    #dvdt = func_crrc.DrawDerivative()

    return func_crrc

def CRRC_Fit(graph, fitmin, fitmax, print_parameters = false):
    func_crrc =  TF1("func_crrc","[4] + [0]*(TMath::Exp(-(x-[3])/[2]) - TMath::Exp(-(x-[3])/[1]))", fitmin,fitmax)
  
    func_crrc.SetLineColor(kBlack)
    func_crrc.SetNpx(1500) 
    func_crrc.SetParameter(0,-9)
    func_crrc.SetParameter(1,0.01)
    func_crrc.SetParameter(2,.55)
    func_crrc.SetParameter(3,0.01)
    func_crrc.SetParameter(4,-0.1)

    graph.Fit(func_crrc,"","",fitmin,fitmax)
    func_crrc.Draw("same")
    
    print "Time Offset = ", str(func_crrc.GetParameter(3))
    print "DC Level = " , str(func_crrc.GetParameter(4))
    print "Amplitude = ", str(func_crrc.GetParameter(0))
    print "Rise time = ", str(func_crrc.GetParameter(2))
    print "Fall time = ", str(func_crrc.GetParameter(1))

    a = func_crrc.GetParameter(0)
    b = func_crrc.GetParameter(1)
    c = func_crrc.GetParameter(2)
    d = func_crrc.GetParameter(3)
    e = func_crrc.GetParameter(4)

    if print_parameters:
        graph.Draw()
        func_crrc.Draw("same")

        textbox = TPaveText(0.7,0.7,1.0,0.55,"NDC")
        #textbox.AddText("Eqn: (([a] / (TMath::Exp (-(x-[d])/[c]) + TMath::Exp ((x-[d])/[b])))) + [e]")
        textbox.AddText(TruncateTo_n_SigFigs(a,3) + " - mV (amplitude)")
        textbox.AddText(TruncateTo_n_SigFigs(b,7) + " - #mus (rise time RC const)")
        textbox.AddText(TruncateTo_n_SigFigs(c,7) + " - #mus (fall time RC const)")
        textbox.AddText(TruncateTo_n_SigFigs(d,7) + " - #mus (trigger offset)")
        textbox.AddText(TruncateTo_n_SigFigs(e,7) + " - mV (DC offset)")
        textbox.Draw("same")




    return func_crrc



def PlotGraphWithErrors(Datafile, rootTitle, Title, xseries_col_num, yseries_col_num, xerrors_col_num, yerrors_col_num, xunits, yunits, skiplines = 0, reverse_xaxis = false ):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    xerrlist, yerrlist = array('d'), array('d')
    linecontents = []
    linecounter = 0
    ymax = 0.

    for line in localfile.readlines()[skiplines:]: #read in data - skip first 5 lines (for header)
        linecontents = line.split("\t")
        x = linecontents[xseries_col_num-1]
        y = linecontents[yseries_col_num-1]
        if (xerrors_col_num!=-1):        
            xerror = linecontents[xerrors_col_num-1]
        else:
            xerror = 0
        if (yerrors_col_num!=-1):
            yerror = linecontents[yerrors_col_num-1]
        else:
            yerror = 0
        linecounter +=1

        if reverse_xaxis:
            xlist.append(-1.*float(x))
            ylist.append(float(y))
            xerrlist.append(-1.*float(xerror))
            yerrlist.append(float(yerror))
        else:
            xlist.append(float(x))
            ylist.append(float(y))
            xerrlist.append(float(xerror))
            yerrlist.append(float(yerror))

        if float(y) > ymax: ymax = float(y) # find max to get the y scale

    
    gr = TGraphErrors(len(xlist), xlist, ylist, xerrlist, yerrlist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.SetMarkerStyle(21)
    #gr.GetXaxis().SetRangeUser(xlist.[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange

    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)
    #gr.Draw("AP")
       
    return gr

def PlotGraphWithErrorsAndWeightings(Datafile, rootTitle, Title, xseries_col_num, yseries_col_num, xerrors_col_num, yerrors_col_num, xweightings_col, xunits, yunits, skiplines = 0, reverse_xaxis = false ):
    
    localfile = open(Datafile) #open file
    xlist, ylist = array('d'), array('d')
    xerrlist, yerrlist = array('d'), array('d')
    linecontents = []
    ymax = 0.

    for line in localfile.readlines()[skiplines:]: #read in data - skip first 5 lines (for header)
        linecontents = line.split("\t")
        x = linecontents[xseries_col_num-1]
        y = linecontents[yseries_col_num-1]
        weight = linecontents[xweightings_col-1]

        if (xerrors_col_num!=-1):        
            xerror = linecontents[xerrors_col_num-1]
        else:
            xerror = 0
        if (yerrors_col_num!=-1):
            yerror = linecontents[yerrors_col_num-1]
        else:
            yerror = 0

        if reverse_xaxis:
            xlist.append(-1.*float(x))
            ylist.append(float(y))
            xerrlist.append(-1.*float(xerror))
            yerrlist.append(float(yerror))
        else:
            if (xweightings_col<>-1):
                for i in range(0,int(weight)):
                    xlist.append(float(x))
                    ylist.append(float(y))
                    xerrlist.append(float(xerror))
                    yerrlist.append(float(yerror))
            else:
                xlist.append(float(x))
                ylist.append(float(y))
                xerrlist.append(float(xerror))
                yerrlist.append(float(yerror))


        if float(y) > ymax: ymax = float(y) # find max to get the y scale

    print "Number of points =",len(xlist)
    gr = TGraphErrors(len(xlist), xlist, ylist, xerrlist, yerrlist) #populate graph with data points

    gr.GetXaxis().SetTitle( xunits )
    gr.GetYaxis().SetTitle( yunits )
    gr.SetMarkerStyle(21)
    #gr.GetXaxis().SetRangeUser(xlist.[0],xlist[-1]) # set x range so graph touches sides, leave y axis to autorange

    SetGraphStyleScopePlot(gr, Title)
    gr.SetName(rootTitle)
    #gr.Draw("AP")
       
    return gr

def PlotHistogramFromScopeTrace(Datafile, nbins, rootTitle, Title, xtitle, nofit=False, force_histmin = 21530176, rescale_x = 1, force_histmax = 321651961, skiplines = 0, abort_after=int(1e9)):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.

    linecontents = []

    for i,line in enumerate(localfile.readlines()[skiplines:abort_after]):
    #for i,line in enumerate(localfile.readlines()[skiplines:]):
        #linecontents = line.split("\t")
        linecontents = line.split(",")
        
        if len(linecontents) == 3: #deals with different pulse height file formats
            value = rescale_x * float(linecontents[2])
        elif len(linecontents) == 1:
            if (linecontents[0] == "NaN\n"): continue
            value = rescale_x * float(linecontents[1])
        else:
            value = rescale_x * float(linecontents[1])

        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value
        
        #if (i % 10000 == 0):
        #    print "Read in",i ,"lines so far"

    #for line in localfile.readlines()[skiplines:]:
    #    linecontents = line.split("\t")
    #    
    #    if len(linecontents) == 3: #deals with different pulse height file formats
    #        value = rescale_x * float(linecontents[2])
    #    elif len(linecontents) == 1:
    #        if (linecontents[0] == "NaN\n"): continue
    #        value = rescale_x * float(linecontents[0])
    #    else:
    #        value = rescale_x * float(linecontents[0])
    #        
    #    datalist.append(value)
    #    if value < minval: minval = value
    #    if value > maxval: maxval = value

    #if (force_histmin <> 21530176):
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-force_histmin))), force_histmin, maxval)
    #else:
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)
    if (force_histmin <> 21530176) and (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, force_histmax)
    elif (force_histmin <> 21530176):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, maxval)
    elif (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, minval, force_histmax)
    else:
        temp_hist = TH1F(rootTitle, Title, nbins, minval, maxval)

    #print ("Minval = " + str(minval))
    #print ("max = " + str(maxval))
    
    for value in datalist:
        temp_hist.Fill(value)

    #temp_hist.Fit("landau", "", "")
    #temp_hist.Fit("landau", "", "",  200, 500)
    #temp_hist.Fit("landau", "", "",  150, 1500)
    temp_hist.GetXaxis().SetTitle(xtitle)
    temp_hist.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    if (nofit == False):
        #fitfunc = TF1(temp_hist.GetFunction("gaus"))
        fitfunc = TF1("gaus", "gaus")
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        temp_hist.Fit(fitfunc)
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        mean = fitfunc.GetParameter(1)
        RMS = fitfunc.GetParameter(2)
    
    if (GRAPHS_BLACK_AND_WHITE):
        temp_hist.SetLineColor(kBlack)

    localfile.close()
    
    return temp_hist, mean, RMS

def AddDataToHistogramFromScopeTrace(Hitogram, Datafile, nbins, rootTitle, Title, xtitle, nofit=False, force_histmin = 21530176, rescale_x = 1, force_histmax = 321651961, skiplines = 0, abort_after=int(1e9)):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.

    linecontents = []

    for i,line in enumerate(localfile.readlines()[skiplines:abort_after]):
    #for i,line in enumerate(localfile.readlines()[skiplines:]):
        #linecontents = line.split("\t")
        linecontents = line.split(",")
        
        if len(linecontents) == 3: #deals with different pulse height file formats
            value = rescale_x * float(linecontents[2])
        elif len(linecontents) == 1:
            if (linecontents[0] == "NaN\n"): continue
            value = rescale_x * float(linecontents[1])
        else:
            value = rescale_x * float(linecontents[1])

        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value
        
        #if (i % 10000 == 0):
        #    print "Read in",i ,"lines so far"

    #for line in localfile.readlines()[skiplines:]:
    #    linecontents = line.split("\t")
    #    
    #    if len(linecontents) == 3: #deals with different pulse height file formats
    #        value = rescale_x * float(linecontents[2])
    #    elif len(linecontents) == 1:
    #        if (linecontents[0] == "NaN\n"): continue
    #        value = rescale_x * float(linecontents[0])
    #    else:
    #        value = rescale_x * float(linecontents[0])
    #        
    #    datalist.append(value)
    #    if value < minval: minval = value
    #    if value > maxval: maxval = value

    #if (force_histmin <> 21530176):
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-force_histmin))), force_histmin, maxval)
    #else:
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)
    if (force_histmin <> 21530176) and (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, force_histmax)
    elif (force_histmin <> 21530176):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, maxval)
    elif (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, minval, force_histmax)
    else:
        temp_hist = TH1F(rootTitle, Title, nbins, minval, maxval)

    #print ("Minval = " + str(minval))
    #print ("max = " + str(maxval))
    
    for value in datalist:
        Hitogram.Fill(value)

    #temp_hist.Fit("landau", "", "")
    #temp_hist.Fit("landau", "", "",  200, 500)
    #temp_hist.Fit("landau", "", "",  150, 1500)
    Hitogram.GetXaxis().SetTitle(xtitle)
    Hitogram.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    if (nofit == False):
        #fitfunc = TF1(temp_hist.GetFunction("gaus"))
        fitfunc = TF1("gaus", "gaus")
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        Hitogram.Fit(fitfunc)
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        mean = fitfunc.GetParameter(1)
        RMS = fitfunc.GetParameter(2)
    
    if (GRAPHS_BLACK_AND_WHITE):
        Hitogram.SetLineColor(kBlack)

    localfile.close()
    
    return Hitogram, mean, RMS

def PlotHistogram_CHARGE_ON_X_AXIS(Datafile, nbins, rootTitle, Title, xtitle, nofit=False, force_histmin = 21530176, rescale_x = 1, force_histmax = 321651961, skiplines = 0):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.

    linecontents = []

    for i,line in enumerate(localfile.readlines()[skiplines:]):
    #for i,line in enumerate(localfile.readlines()[skiplines:]):
        linecontents = line.split("\t")

        
        if len(linecontents) == 3: #deals with different pulse height file formats
            if (linecontents[2] == "NaN\n"): continue
            value = rescale_x  * (36./1000.) * float(linecontents[2])
        elif len(linecontents) == 1:
            if (linecontents[0] == "NaN\n"): continue
            value = rescale_x  * (36./1000.) * float(linecontents[0])
        else:
            value = rescale_x  * (36./1000.) * float(linecontents[0])

        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value
        
        if (i % 10000 == 0):
            print "Read in",i ,"lines so far"

    #for line in localfile.readlines()[skiplines:]:
    #    linecontents = line.split("\t")
    #    
    #    if len(linecontents) == 3: #deals with different pulse height file formats
    #        value = rescale_x * float(linecontents[2])
    #    elif len(linecontents) == 1:
    #        if (linecontents[0] == "NaN\n"): continue
    #        value = rescale_x * float(linecontents[0])
    #    else:
    #        value = rescale_x * float(linecontents[0])
    #        
    #    datalist.append(value)
    #    if value < minval: minval = value
    #    if value > maxval: maxval = value

    #if (force_histmin <> 21530176):
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-force_histmin))), force_histmin, maxval)
    #else:
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)
    if (force_histmin <> 21530176) and (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin * (36./1000.), force_histmax * (36./1000.))
    elif (force_histmin <> 21530176):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin * (36./1000.), maxval * (36./1000.))
    elif (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, minval * (36./1000.), force_histmax * (36./1000.))
    else:
        temp_hist = TH1F(rootTitle, Title, nbins, minval * (36./1000.), maxval * (36./1000.))

    #print ("Minval = " + str(minval))
    #print ("max = " + str(maxval))
    
    for value in datalist:
        #temp_hist.Fill(-1.*value)
        temp_hist.Fill(value)

    #temp_hist.Fit("landau", "", "")
    #temp_hist.Fit("landau", "", "",  200, 500)
    #temp_hist.Fit("landau", "", "",  150, 1500)
    if str.find(xtitle,"#mum")<>-1:
        temp_hist.GetXaxis().SetTitle("Charge (ke^{-})")
    else:
        temp_hist.GetXaxis().SetTitle(xtitle)

    temp_hist.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    if (nofit == False):
        #fitfunc = TF1(temp_hist.GetFunction("gaus"))
        fitfunc = TF1("gaus", "gaus")
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        temp_hist.Fit(fitfunc)
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        mean = fitfunc.GetParameter(1)
        RMS = fitfunc.GetParameter(2)
    
    if (GRAPHS_BLACK_AND_WHITE):
        temp_hist.SetLineColor(kBlack)

    localfile.close()

    temp_hist.GetYaxis().SetTitleOffset(1.2)
    
    return temp_hist, mean, RMS

def PlotHistogram(Datafile, nbins, rootTitle, Title, xtitle, nofit=False, force_histmin = 21530176, rescale_x = 1, force_histmax = 321651961, skiplines = 0):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.

    linecontents = []

    for i,line in enumerate(localfile.readlines()[skiplines:]):
    #for i,line in enumerate(localfile.readlines()[skiplines:]):
        linecontents = line.split("\t")

        
        if len(linecontents) == 3: #deals with different pulse height file formats
            if (linecontents[2] == "NaN\n"): continue
            value = rescale_x  * float(linecontents[2])
        elif len(linecontents) == 1:
            if (linecontents[0] == "NaN\n"): continue
            value = rescale_x  * float(linecontents[0])
        else:
            value = rescale_x  * float(linecontents[0])

        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value
        
        if (i % 10000 == 0):
            print "Read in",i ,"lines so far"

    #for line in localfile.readlines()[skiplines:]:
    #    linecontents = line.split("\t")
    #    
    #    if len(linecontents) == 3: #deals with different pulse height file formats
    #        value = rescale_x * float(linecontents[2])
    #    elif len(linecontents) == 1:
    #        if (linecontents[0] == "NaN\n"): continue
    #        value = rescale_x * float(linecontents[0])
    #    else:
    #        value = rescale_x * float(linecontents[0])
    #        
    #    datalist.append(value)
    #    if value < minval: minval = value
    #    if value > maxval: maxval = value

    #if (force_histmin <> 21530176):
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-force_histmin))), force_histmin, maxval)
    #else:
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)
    if (force_histmin <> 21530176) and (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, force_histmax)
    elif (force_histmin <> 21530176):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin , maxval )
    elif (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, minval , force_histmax )
    else:
        temp_hist = TH1F(rootTitle, Title, nbins, minval , maxval )

    #print ("Minval = " + str(minval))
    #print ("max = " + str(maxval))
    
    for value in datalist:
        temp_hist.Fill(value)

    #temp_hist.Fit("landau", "", "")
    #temp_hist.Fit("landau", "", "",  200, 500)
    #temp_hist.Fit("landau", "", "",  150, 1500)
    temp_hist.GetXaxis().SetTitle(xtitle)
    temp_hist.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    if (nofit == False):
        #fitfunc = TF1(temp_hist.GetFunction("gaus"))
        fitfunc = TF1("gaus", "gaus")
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        temp_hist.Fit(fitfunc)
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        mean = fitfunc.GetParameter(1)
        RMS = fitfunc.GetParameter(2)
    
    if (GRAPHS_BLACK_AND_WHITE):
        temp_hist.SetLineColor(kBlack)

    localfile.close()

    temp_hist.GetYaxis().SetTitleOffset(1.2)
    
    return temp_hist, mean, RMS

def PlotHistogram_temp(Datafile, nbins, rootTitle, Title, xtitle, nofit=False, force_histmin = 21530176, rescale_x = 1, force_histmax = 321651961, skiplines = 0, max_pts = 3000000000):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.
    sum = 0.
    npts = 1.

    linecontents = []

    for i,line in enumerate(localfile.readlines()[skiplines:]):
        linecontents = line.split("\t")
        
        if len(linecontents) == 3: #deals with different pulse height file formats
            value = rescale_x * float(linecontents[2])
        elif len(linecontents) == 1:
            if (linecontents[0] == "NaN\n"): continue
            value = rescale_x * float(linecontents[0])
        else:
            value = rescale_x * float(linecontents[0])

        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value
        
        if (value > 50):
            sum += value
            npts += 1

        if (i % 10000 == 0):
            print "Read in",i ,"lines so far"
        if (i == max_pts - 1):
            break

    #for line in localfile.readlines()[skiplines:]:
    #    linecontents = line.split("\t")
    #    
    #    if len(linecontents) == 3: #deals with different pulse height file formats
    #        value = rescale_x * float(linecontents[2])
    #    elif len(linecontents) == 1:
    #        if (linecontents[0] == "NaN\n"): continue
    #        value = rescale_x * float(linecontents[0])
    #    else:
    #        value = rescale_x * float(linecontents[0])
    #        
    #    datalist.append(value)
    #    if value < minval: minval = value
    #    if value > maxval: maxval = value

    #if (force_histmin <> 21530176):
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-force_histmin))), force_histmin, maxval)
    #else:
    #    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)
    if (force_histmin <> 21530176) and (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, force_histmax)
    elif (force_histmin <> 21530176):
        temp_hist = TH1F(rootTitle, Title, nbins, force_histmin, maxval)
    elif (force_histmax <> 321651961):
        temp_hist = TH1F(rootTitle, Title, nbins, minval, force_histmax)
    else:
        temp_hist = TH1F(rootTitle, Title, nbins, minval, maxval)

    #print ("Minval = " + str(minval))
    #print ("max = " + str(maxval))
    
    for value in datalist:
        temp_hist.Fill(value)

    #temp_hist.Fit("landau", "", "")
    #temp_hist.Fit("landau", "", "",  200, 500)
    #temp_hist.Fit("landau", "", "",  150, 1500)
    temp_hist.GetXaxis().SetTitle(xtitle)
    temp_hist.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    if (nofit == False):
        #fitfunc = TF1(temp_hist.GetFunction("gaus"))
        fitfunc = TF1("gaus", "gaus")
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        temp_hist.Fit(fitfunc)
        if (GRAPHS_BLACK_AND_WHITE):
            fitfunc.SetLineColor(kBlack)
        #mean = fitfunc.GetParameter(1)
        RMS = fitfunc.GetParameter(2)
    mean = sum/npts

    if (GRAPHS_BLACK_AND_WHITE):
        temp_hist.SetLineColor(kBlack)
    
    return temp_hist, mean, RMS
def PlotNegativeHistogram(Datafile, nbins, rootTitle, Title, xtitle):
    localfile = open(Datafile)
    datalist = array('d')
   
    minval = 1e9
    maxval = -1e9
    value = 0.

    linecontents = []

    for line in localfile.readlines()[1:]:
        linecontents = line.split("\t")
        
        if len(linecontents) == 3: #deals with different pulse height file formats
            value = -1.0 * float(linecontents[2])
        elif len(linecontents) == 1:
            value = -1.0 * float(linecontents[0])
        else:
            value = -1.0 * float(linecontents[0])
            
        datalist.append(value)
        if value < minval: minval = value
        if value > maxval: maxval = value

    temp_hist = TH1F(rootTitle, Title, int(nbins * (maxval/(maxval-minval))), minval, maxval)

    for value in datalist:
        temp_hist.Fill(value)

    #temp_hist.Fit("landau", "", "",  200, 500)
    temp_hist.GetXaxis().SetTitle(xtitle)
    temp_hist.GetYaxis().SetTitle("Counts")
    
    mean = 0.0
    RMS = 0.0
    
    #temp_hist.Fit("gaus")
    #fitfunc = TF1(temp_hist.GetFunction("gaus"))
    #mean = fitfunc.GetParameter(1)
    #RMS = fitfunc.GetParameter(2)
    
    return temp_hist, mean, RMS
def ScaleLandau(TF1_to_scale, TF1_to_scale_to):
    scale_ratio = TF1_to_scale_to.GetMaximum() / TF1_to_scale.GetMaximum()
    TF1_to_scale.SetParameter(0, TF1_to_scale.GetParameter(0) * scale_ratio)
    return 


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


def LanGausPlusGausFit_Fixed_MPV(hist, fitmin, fitmax, MPV_Fix):

    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(fitmin*CHARGE_ADJUSTMENT)
    fitrange.append(fitmax*CHARGE_ADJUSTMENT)

    print ("Integral = " + str(hist.GetSum()))

    initial_pars.append(hist.GetRMS()/7)
    initial_pars.append(MPV_Fix)
    initial_pars.append(1200000)
    initial_pars.append(29.8*CHARGE_ADJUSTMENT)
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




def LanGausPlusGausFit_Poly(hist, fitmin, fitmax):

    fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr = array('d'), array('d'), array('d'), array('d'), array('d'), array('d'), array('d')
    ret_NDF = array('i')

    LINE_WIDTH = 5

    fitrange.append(fitmin*CHARGE_ADJUSTMENT)
    fitrange.append(fitmax*CHARGE_ADJUSTMENT)

    integral = hist.GetSum()

    print "Integral = " , integral

    initial_pars.append(hist.GetRMS()/7)
    initial_pars.append(hist.GetBinCenter(hist.GetMaximumBin()))
    initial_pars.append(integral)
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
    initial_pars.append(150)#noise gaus height
    initial_pars.append(-0.1*CHARGE_ADJUSTMENT)#noise gaus meanasdasdasd
    initial_pars.append(2.*CHARGE_ADJUSTMENT)#noise gaus sigma
    parlimits_lo.append(0)
    parlimits_lo.append(-1.)
    parlimits_lo.append(-3.)
    parlimits_hi.append(0)
    parlimits_hi.append(1.)
    parlimits_hi.append(3.)
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

def LanGausPlusGaus_AllFilesInDir(input_dir, outfile_name, histmin, histmax, fitmin, fitmax, nbins):
    import os

    files = os.listdir(input_dir)

    amplitude_list = []

    for currentfile in files:
        filename = input_dir + currentfile
        
        if (currentfile[len(currentfile)-3:] != "dat"):
            print "Skipping " + str(currentfile)
            continue

        c1 = TCanvas( '', '', 500, 200, 700, 500 ) #create canvas
        SetCanvasStyle(c1)

        hist, dummy, dummy2 = PlotHistogram_CHARGE_ON_X_AXIS(filename, nbins,"Pumped wafer point","", "CCD of pulse (#mum)", true, histmin, CCD_Gain_factor, histmax)

        fitfunc = LanGausPlusGausFit(hist, fitmin, fitmax)
        hist.Draw()
        fitfunc.Draw("same")
        c1.SaveAs(filename[:-4] + "_refit.pdf")

        CCD = str(fitfunc.GetParameter(1))
        CCD_error = str(fitfunc.GetParError(1))

        FWHM, FWHM_L, FWHM_R = GetFWHM(hist)
        
        outfile = open(outfile_name,'a')
        outfile.write(str(CCD) + "\t" + str(CCD_error) + "\t" + str(FWHM) + "\t" + str(FWHM_L) + "\t"+ str(FWHM_R) + "\n")
        outfile.close

        print "Just finished processing file: " + currentfile
        del c1, hist, dummy, dummy2, fitfunc


def LanGausPlusGaus_AllFilesInDir_Hysteresis(input_dir, outfile_name, histmin, histmax, fitmin, fitmax, nbins):
    import os

    files = os.listdir(input_dir)

    amplitude_list = []

    for currentfile in files:
        filename = input_dir + currentfile + "\\data\\PHA_0.dat"
        #
        #if (currentfile[len(currentfile)-3:] != "dat"):
        #    print "Skipping " + str(currentfile)
        #    continue

        c1 = TCanvas( '', '', 500, 200, 700, 500 ) #create canvas
        SetCanvasStyle(c1)

        hist, dummy, dummy2 = PlotHistogram_CHARGE_ON_X_AXIS(filename, nbins,"Pumped wafer point","", "CCD of pulse (#mum)", true, histmin, CCD_Gain_factor, histmax)

        fitfunc = LanGausPlusGausFit(hist, fitmin, fitmax)
        hist.Draw()
        fitfunc.Draw("same")
        c1.SaveAs(input_dir + currentfile + "_refit_up.pdf")

        CCD = str(fitfunc.GetParameter(1))
        CCD_error = str(fitfunc.GetParError(1))

        FWHM, FWHM_L, FWHM_R = GetFWHM(hist)
        
        outfile = open(outfile_name,'a')
        outfile.write(currentfile + "\t" + str(CCD) + "\t" + str(CCD_error) + "\t" + str(FWHM) + "\t" + str(FWHM_L) + "\t"+ str(FWHM_R) + "\n")
        outfile.close

        print "Just finished processing file: " + currentfile
        del c1, hist, dummy, dummy2, fitfunc


        
def LanGausPlusGaus_AllFilesInDir_Pumping(input_dir, outfile_name, histmin, histmax, fitmin, fitmax, nbins):
    import os

    files = os.listdir(input_dir)

    amplitude_list = []

    for currentfile in files:
        if (str.count(currentfile,"."))<>0:
            print "Skipping " + str(currentfile)
            continue
        
        filename = input_dir + currentfile + "\\data\\PHA_0.dat"


        c1 = TCanvas( '', '', 500, 200, 700, 500 ) #create canvas
        SetCanvasStyle(c1)

        hist, dummy, dummy2 = PlotHistogram_CHARGE_ON_X_AXIS(filename, nbins,"Pumped wafer point","", "CCD of pulse (#mum)", true, histmin, CCD_Gain_factor, histmax)

        fitfunc = LanGausPlusGausFit(hist, fitmin, fitmax)
        hist.Draw()
        fitfunc.Draw("same")
        c1.SaveAs(input_dir + currentfile + "_refit_up.pdf")

        CCD = str(fitfunc.GetParameter(1))
        CCD_error = str(fitfunc.GetParError(1))

        FWHM, FWHM_L, FWHM_R = GetFWHM(hist)
        
        outfile = open(outfile_name,'a')
        outfile.write(currentfile + "\t" + str(CCD) + "\t" + str(CCD_error) + "\t" + str(FWHM) + "\t" + str(FWHM_L) + "\t"+ str(FWHM_R) + "\n")
        outfile.close

        print "Just finished processing file: " + currentfile
        del c1, hist, dummy, dummy2, fitfunc

        
def LanGausPlusGaus_AllFilesInDir_Pumping_Poly(input_dir, outfile_name, histmin, histmax, fitmin, fitmax, nbins):
    import os

    files = os.listdir(input_dir)

    amplitude_list = []

    for currentfile in files:
        if (str.count(currentfile,"."))<>0:
            print "Skipping " + str(currentfile)
            continue
        
        filename = input_dir + currentfile + "\\data\\PHA_0.dat"


        c1 = TCanvas( '', '', 500, 200, 700, 500 ) #create canvas
        SetCanvasStyle(c1)

        hist, dummy, dummy2 = PlotHistogram_CHARGE_ON_X_AXIS(filename, nbins,"Pumped wafer point","", "CCD of pulse (#mum)", true, histmin, CCD_Gain_factor, histmax)

        #fitfunc = LanGausFit(hist, fitmin, fitmax)
        fitfunc = LanGausPlusGausFit(hist, fitmin, fitmax)
        hist.Draw()
        fitfunc.Draw("same")
        c1.SaveAs(input_dir + currentfile + "_refit_up.pdf")

        CCD = str(fitfunc.GetParameter(1))
        CCD_error = str(fitfunc.GetParError(1))

        FWHM, FWHM_L, FWHM_R = GetFWHM(hist)
        
        outfile = open(outfile_name,'a')
        outfile.write(currentfile + "\t" + str(CCD) + "\t" + str(CCD_error) + "\n")
        outfile.close

        print "Just finished processing file: " + currentfile
        del c1, hist, dummy, dummy2, fitfunc

def LandauFit(hist, fitmin, fitmax):

    fitfunc = TF1("landfunc","landau",fitmin,fitmax)
    fitfunc.SetNpx(10000)
    hist.Fit(fitfunc)
    return fitfunc

def LandauFit_temp(hist, fitmin, fitmax):

    fitfunc = TF1("landfunc","landau",fitmin,fitmax)
    fitfunc.SetNpx(1000)
    fitfunc.SetLineColor(kGreen)
    hist.Fit(fitfunc)
    return fitfunc

def LanGausFit(hist, fitmin, fitmax):


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

    print ("Initial pars:\n")
    print ("par0 = " + str(initial_pars[0]))
    print ("par1 = " + str(initial_pars[1]))
    print ("par2 = " + str(initial_pars[2]))
    print ("par3 = " + str(initial_pars[3]))


    convolfunc = ROOT.langaufit(hist,fitrange, initial_pars, parlimits_lo, parlimits_hi, ret_pars, ret_errors, ret_ChiSqr, ret_NDF)
    convolfunc.SetNpx(1000)
    convolfunc.SetLineStyle(1)
    convolfunc.SetLineColor(kRed)
    convolfunc.SetLineWidth(LINE_WIDTH)


    try:
        chisqr_over_NDF = ret_ChiSqr[0]/ret_NDF[0]
    except:
        chisqr_over_NDF = -1
    #chisqr_over_NDF = ret_ChiSqr[0]/ ret_NDF[0]#this can be div by zero hence try catch above

    print ("Chisqr = " + str(ret_ChiSqr[0]))
    print ("NDF = " + str(ret_NDF[0]))
    print ("Chisqr / NDF = " + str(chisqr_over_NDF))

    #hist.Draw("")
    #convolfunc.Draw("same")

    print ("Final pars:\n")
    print ("par0 = " + str(ret_pars[0]))
    print ("par1 = " + str(ret_pars[1]))
    print ("par2 = " + str(ret_pars[2]))
    print ("par3 = " + str(ret_pars[3]))



    return convolfunc

def VavilovFit(hist, fitmin, fitmax, beta):

    #fitfunc = TF1("vavilov_function","[1]*(TMath::Vavilov([2]*x - [0],[3],[4]))",  fitmin, fitmax)
    #fitfunc.SetParameters(11.3,22000,0.02, 0.01, beta)

    fitfunc = TF1("vavilov_function","[4]*(TMath::Vavilov((x - [2])/[3],[0],[1]))",  fitmin*CHARGE_ADJUSTMENT, fitmax*CHARGE_ADJUSTMENT)
    fitfunc.SetParameters(0.02,0.96,550*CHARGE_ADJUSTMENT,40*CHARGE_ADJUSTMENT,5000)
    
    fitfunc.SetParLimits(0,0.01,1)
    fitfunc.FixParameter(1,beta)


    fitfunc.SetNpx(10000)
    fitfunc.SetLineStyle(1)
    fitfunc.SetLineColor(kRed)
    fitfunc.SetLineWidth(5)
    
    hist.Fit(fitfunc, "", "", fitmin*CHARGE_ADJUSTMENT, fitmax*CHARGE_ADJUSTMENT)

    print "kappa =",fitfunc.GetParameter(0)

    try:
        chisq = fitfunc.GetChisquare()
        NDF = fitfunc.GetNDF()
        chisqr_over_NDF = chisq/NDF
    except:
        chisqr_over_NDF = -1

    print "Chisqr =", chisq
    print "NDF =", NDF 
    print "Chisqr / NDF =",chisqr_over_NDF

    #print "\n\n\n\n"
    #print "par0 =", fitfunc.GetParameter(0)
    #print "par1 =", fitfunc.GetParameter(1)
    #print "par2 =", fitfunc.GetParameter(2)
    #print "par3 =", fitfunc.GetParameter(3)
    #print "par4 =", fitfunc.GetParameter(5)
    
    #hist.Draw("")
    #fitfunc.Draw("same")

    return fitfunc

def VavilovFit_FixedKappa(hist, fitmin, fitmax, beta, kappa):

    #fitfunc = TF1("vavilov_function","[1]*(TMath::Vavilov([2]*x - [0],[3],[4]))",  fitmin, fitmax)
    #fitfunc.SetParameters(11.3,22000,0.02, 0.01, beta)

    fitfunc = TF1("vavilov_function","[4]*(TMath::Vavilov((x - [2])/[3],[0],[1]))",  fitmin*CHARGE_ADJUSTMENT, fitmax*CHARGE_ADJUSTMENT)
    fitfunc.SetParameters(0.02,0.96,550,40,1000)
    
    fitfunc.SetParLimits(0,0.01,1)
    fitfunc.FixParameter(1,beta)

    fitfunc.SetParLimits(4,0.0000001,1e9)
    fitfunc.FixParameter(0,kappa)


    fitfunc.SetNpx(10000)
    fitfunc.SetLineStyle(1)
    fitfunc.SetLineColor(kRed)
    fitfunc.SetLineWidth(5)
    
    hist.Fit(fitfunc, "", "", fitmin*CHARGE_ADJUSTMENT, fitmax*CHARGE_ADJUSTMENT)

    print "kappa =",fitfunc.GetParameter(0)

    try:
        chisq = fitfunc.GetChisquare()
        NDF = fitfunc.GetNDF()
        chisqr_over_NDF = chisq/NDF
    except:
        chisqr_over_NDF = -1

    print "Chisqr =", chisq
    print "NDF =", NDF 
    print "Chisqr / NDF =",chisqr_over_NDF

    #print "\n\n\n\n"
    #print "par0 =", fitfunc.GetParameter(0)
    #print "par1 =", fitfunc.GetParameter(1)
    #print "par2 =", fitfunc.GetParameter(2)
    #print "par3 =", fitfunc.GetParameter(3)
    #print "par4 =", fitfunc.GetParameter(5)
    
    #hist.Draw("")
    #fitfunc.Draw("same")

    return fitfunc

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
    filename = "./ROOT_Browserfile.root"
    ROOTfile = TFile.Open(filename, "RECREATE")
    object.Write()
    ROOTfile.Close()
    Popen('cmd.exe /C start %s' %filename)
def Finally():
    print("\n\n*******\nEnd of code\n*******")
    sleep(PAUSE_AT_END)

if __name__=="__main__": # main code block
    

    print "You are running the wrong file!"
    print "End of python code"
    Finally()
