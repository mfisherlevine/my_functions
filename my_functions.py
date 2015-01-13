from _ctypes import Array
from numpy import array
from __builtin__ import open

intrinsic_offset = -75

# def TimepixToExposure(filename):
#     from lsst.afw.image import makeImageFromArray
#     import numpy as np
#     
#     data = np.loadtxt(filename)
#     x = data[:, 0] 
#     y = data[:, 1] 
#     t = data[:, 2]
# 
#     my_array = np.zeros((256,256), dtype = np.int32)
# 
#     for pointnum in range(len(x)):
#         my_array[x[pointnum],y[pointnum]] = t[pointnum]
#     
#     my_image = makeImageFromArray(my_array)
#     return my_image

def TimepixToExposure(filename, xmin, xmax, ymin, ymax):
    import numpy as np
    from lsst.afw.image import makeImageFromArray

    data = np.loadtxt(filename)

    my_array = np.zeros((256,256), dtype = np.int32)
    
    if data.shape == (0,):
        my_image = makeImageFromArray(my_array)
        
    elif data.shape == (3,):
        x = data[0] 
        y = data[1] 
        t = data[2]
        
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            my_array[y,x] = t
      
        my_image = makeImageFromArray(my_array)
    
    else:   
        x = data[:, 0] 
        y = data[:, 1] 
        t = data[:, 2]
    
        for pointnum in range(len(x)):
            if x[pointnum] >= xmin and x[pointnum] <= xmax and y[pointnum] >= ymin and y[pointnum] <= ymax:
                my_array[y[pointnum],x[pointnum]] = t[pointnum]
            
        my_image = makeImageFromArray(my_array)
    
    return my_image




def XYI_array_to_exposure(xs, ys, i_s):
    from lsst.afw.image import makeImageFromArray
    import numpy as np

    my_array = np.zeros((256,256), dtype = np.int32)

    for pointnum in range(len(xs)):
        my_array[xs[pointnum],ys[pointnum]] = i_s[pointnum]
    
    my_image = makeImageFromArray(my_array)
    return my_image



def GetTimecodes_SingleFile(filename, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0):
    import string
    
    timecodes = []
    datafile = open(filename)
    
    for line in datafile.readlines():
        x,y,timecode = string.split(str(line),'\t')
        x = int(x)
        y = int(y)
        timecode = int(timecode)
        if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
            actual_offset_us = intrinsic_offset - offset_us
            time_s = (11810. - timecode) * 20e-9
            time_us = (time_s *1e6)- actual_offset_us
            timecodes.append(time_us)
    
    return timecodes



def GetRawTimecodes_SingleFile(filename, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0):
    import string
    
    timecodes = []
    datafile = open(filename)
    
    for line in datafile.readlines():
        x,y,timecode = string.split(str(line),'\t')
        x = int(x)
        y = int(y)
        timecode = int(timecode)
        if x >= winow_xmin and x <= winow_xmax and y >= winow_ymin and y <= winow_ymax: 
            if timecode <> 11810:
                timecodes.append(timecode) 

    return timecodes

def GetTimecodes_AllFilesInDir(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0):
    import string, os
    
    timecodes = []
    files = []
    
    for filename in os.listdir(path):
        files.append(path + filename)

    for filename in files:
        datafile = open(filename)
        lines = datafile.readlines()
        
#         if len(lines) > 50 and len(lines) < 1000:
#             OpenTimepixInDS9(filename)
#             exit()
        
        if len(lines) > 1000: continue #skip bad files (glitches)
        for line in lines:
            x,y,timecode = string.split(str(line),'\t')
            x = int(x)
            y = int(y)
            timecode = int(timecode)
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                actual_offset_us = intrinsic_offset - offset_us
                time_s = (11810. - timecode) * 20e-9
                time_us = (time_s *1e6)- actual_offset_us
                timecodes.append(time_us)

    return timecodes

def GetXYTarray_AllFilesInDir(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, tmin_us = -1000, tmax_us = 999999, maxfiles = None):
    import string, os
    import pylab as pl
    
    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    xs, ys, ts = [], [], []

    num = 0
    for filename in files:
        data = pl.loadtxt(filename, usecols = (0,1,2))
        num +=1
        if (num % 10 == 0): print 'loaded %s files'%num
        
        #handle problem with the way loadtxt reads single line data files
        if data.shape == (3,): 
            x = int(data[0])
            y = int(data[1])
            timecode = int(data[2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                actual_offset_us = intrinsic_offset - offset_us
                time_s = (11810. - timecode) * 20e-9
                time_us = (time_s *1e6)- actual_offset_us
                if time_us>=tmin_us and time_us<= tmax_us:
                    xs.append(x)
                    ys.append(y)
                    ts.append(time_us)
            continue
        
        #extract data for multiline files
        if len(data) > 10000: continue #skip glitch files
        for i in range(len(data)):
            x = int(data[i,0])
            y = int(data[i,1])
            timecode = int(data[i,2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                actual_offset_us = intrinsic_offset - offset_us
                time_s = (11810. - timecode) * 20e-9
                time_us = (time_s *1e6)- actual_offset_us
                if time_us>=tmin_us and time_us<= tmax_us:
                    xs.append(x)
                    ys.append(y)
                    ts.append(time_us)
        
        if maxfiles != None and num == maxfiles:
            return xs, ys, ts

    return xs, ys, ts  


def MakeCompositeImage_Medipix(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, maxfiles = None):
    from lsst.afw.image import makeImageFromArray
    import string, os
    import numpy as np
    import pylab as pl
    
    my_array = np.zeros((256,256), dtype = np.int32)

    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    num = 0
    for filename in files:
        data = pl.loadtxt(filename, usecols = (0,1,2))
        num +=1
        if (num % 10 == 0): print 'loaded %s files'%num
        
        for i in range(len(data)):
            x = int(data[i,0])
            y = int(data[i,1])
            intensity = int(data[i,2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                my_array[x,y] += intensity
      
      
        if maxfiles != None and num == maxfiles:
            my_image = makeImageFromArray(my_array)
            return my_image
    
    my_image = makeImageFromArray(my_array)
    return my_image


def OpenTimepixInDS9(filename):
    import lsst.afw.display.ds9 as ds9
    image = TimepixToExposure(filename, 0,255,0,255)

    try:
        ds9.initDS9(False)
    except ds9.Ds9Error:
        print 'DS9 launch bug error thrown away (probably)'

    ds9.mtv(image)
    
def BuildMosaic(filename, gutter = 0, background = 0):
    import lsst.afw.display.utils as Util
    import lsst.afw.image as afwImg
   
    m = Util.Mosaic()
    m.setGutter(gutter)
    m.setBackground(background)
        
    images = []
    for i in range(2,18):
        m.append(afwImg.ImageF(filename,i), str(i))
    
       
#    mosaic = m.makeMosaic()

    return m


def GetXYTarray_SingleFile(filename, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999):
    import pylab as pl
    
    xs, ys, ts = [], [], []

    data = pl.loadtxt(filename, usecols = (0,1,2))
    
    #handle problem with the way loadtxt reads single line data files
    if data.shape == (3,): 
        x = int(data[0])
        y = int(data[1])
        timecode = int(data[2])
        if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
            xs.append(x)
            ys.append(y)
            ts.append(timecode)
        return xs, ys, ts  
        
    
    #extract data for multiline files
    for i in range(len(data)):
        x = int(data[i,0])
        y = int(data[i,1])
        timecode = int(data[i,2])
        if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
            xs.append(x)
            ys.append(y)
            ts.append(timecode)
    
    return xs, ys, ts


def Timepix_ToT_to_lego(datafile, center_x, center_y, boxsize_over_2, savefile = '', mask_above = 999999, print_RMS = False, fix_zmax = '', nfiles_for_camera_tricks = '', filenum_for_camera_trick = ''):
    from ROOT import TH2F, TCanvas
    from root_functions import CANVAS_HEIGHT, CANVAS_WIDTH
    c1 = TCanvas( 'canvas', 'canvas', CANVAS_WIDTH/2,CANVAS_HEIGHT/2)

    xs, ys, ts = GetXYTarray_SingleFile(datafile, center_x - boxsize_over_2, center_x + boxsize_over_2, center_y - boxsize_over_2, center_y + boxsize_over_2)
    
    nx,ny = 256,256
    image_hist = TH2F('', '',nx,0,255,ny, 0, 255)
    
    for i in range(len(xs)):
        value = float(ts[i])/50
        if value > mask_above: value = 0
        image_hist.Fill(float(xs[i]),float(ys[i]),value)

    image_hist.GetXaxis().SetTitle('x')
    image_hist.GetYaxis().SetTitle('y')
    image_hist.GetZaxis().SetTitle('ToT (us)')
    
    image_hist.GetXaxis().SetRangeUser(center_x - boxsize_over_2, center_x + boxsize_over_2)
    image_hist.GetYaxis().SetRangeUser(center_y - boxsize_over_2, center_y + boxsize_over_2)
    if fix_zmax != '':
        image_hist.GetZaxis().SetRangeUser(0,fix_zmax)
    
    image_hist.GetXaxis().SetTitleOffset(1.2)
    image_hist.GetYaxis().SetTitleOffset(1.4)
    image_hist.GetZaxis().SetTitleOffset(1.2)
    
    image_hist.Draw("same lego2 0 z") #box, lego, colz, lego2 0
    image_hist.SetStats(False)
    
    if savefile != '':
        if nfiles_for_camera_tricks != '' and filenum_for_camera_trick != '':
            c1.SetPhi(180 * filenum_for_camera_trick / nfiles_for_camera_tricks)
        else:
            c1.SetPhi(41.57391)

        c1.SetTheta(41.57391)
#        c1.SetPhi(-132.4635)
        #c1.SetTheta(35)#theta sets inclination, phi sets rotation
#        c1.SetPhi(45)#around z axis
        
        if print_RMS:
            from ROOT import TPaveText
            textbox = TPaveText(0.0,1.0,0.2,0.8,"NDC")
            textbox.AddText('RMS = ' + str(image_hist.GetRMS()))
            textbox.SetFillColor(0)
            textbox.Draw("same")
        c1.SaveAs(savefile)
        
    del c1        
        
    return image_hist
        

def TimepixDirToPImMMSDatafile(path, outfile_name, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, tmin_us = -1000, tmax_us = 999999, maxfiles = None):
    import string, os
    import pylab as pl
    
    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    output_file = open(outfile_name, 'w')

    for filenum, filename in enumerate(files):
        data = pl.loadtxt(filename, usecols = (0,1,2))
        if (filenum % 100 == 0): print 'loaded %s files'%filenum
        
        #handle problem with the way loadtxt reads single line data files
        if data.shape == (3,): 
            x = int(data[0])
            y = int(data[1])
            timecode = int(data[2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                actual_offset_us = intrinsic_offset - offset_us
#                 time_s = (11810. - timecode) * 20e-9
#                 time_us = (time_s *1e6)- actual_offset_us
                reflected_timecode = 11810 - timecode
#                 if time_us>=tmin_us and time_us<= tmax_us:
                line = str(x) + '\t' + str(y) + '\t' + str(reflected_timecode) + '\t' + str(filenum) + '\t' + '1\n'
                output_file.write(line)
            continue
        
        #extract data for multiline files
        if len(data) > 50: continue #skip glitch files
        for i in range(len(data)):
            x = int(data[i,0])
            y = int(data[i,1])
            timecode = int(data[i,2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                actual_offset_us = intrinsic_offset - offset_us
#                 time_s = (11810. - timecode) * 20e-9
#                 time_us = (time_s *1e6)- actual_offset_us
                reflected_timecode = 11810 - timecode
#                 if time_us>=tmin_us and time_us<= tmax_us:
                line = str(x) + '\t' + str(y) + '\t' + str(reflected_timecode) + '\t' + str(filenum) + '\t' + '1\n'
                output_file.write(line)
#                     xs.append(x)
#                     ys.append(y)
#                     ts.append(time_us)

        if maxfiles != None and num == maxfiles:
            output_file.close()
            return

    output_file.close()
    return
      
        

