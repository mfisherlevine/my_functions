from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import zip
from builtins import str
from builtins import range
from _ctypes import Array
from numpy import array
from builtins import open
import numpy as np

from lsst.afw.image import makeImageFromArray
from time import sleep
from matplotlib.pyplot import ylim
import pylab as plt

import shutil
import os
import filecmp
# import pyfits as pf
import sys
import glob
import functions as fn
# fn = reload(fn)

intrinsic_offset = -75


def GetMdSetFromVisitList(butler, visits, mdKey):
    """Return the unique values of a given registry key for a list of visits."""
    mdSet = []
    failed = 0
    for vis in visits:
        try:
            md = butler.queryMetadata('raw', [mdKey], dataId={'visit': vis})
            mdSet.append(md[0])
        except:
            failed += 1
    if failed:
        print("Failed to find expTimes for %s of %s"%(failed, len(visits)))
    return set(mdSet)

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


# def TimepixToExposure_binary(filename, xmin, xmax, ymin, ymax):
#     import numpy as np
#     from lsst.afw.image import makeImageFromArray
# 
#     data = np.loadtxt(filename)
# 
#     my_array = np.zeros((256,256), dtype = np.int32)
#     
#     if data.shape == (0,):
#         my_image = makeImageFromArray(my_array)
#         
#     elif data.shape == (3,):
#         x = data[0] 
#         y = data[1] 
#         t = data[2]
#         
#         if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
#             my_array[y,x] = 1
#       
#         my_image = makeImageFromArray(my_array)
#     
#     else:   
#         x = data[:, 0] 
#         y = data[:, 1] 
#         t = data[:, 2]
#     
#         for pointnum in range(len(x)):
#             if x[pointnum] >= xmin and x[pointnum] <= xmax and y[pointnum] >= ymin and y[pointnum] <= ymax:
#                 my_array[y[pointnum],x[pointnum]] = 1
#             
#         my_image = makeImageFromArray(my_array)
#     
#     return my_image


def SafeCopy(src, dest):
    if os.path.exists(dest):
        print('warning, tried to overwrite %s with %s'%(dest, src))
        if filecmp.cmp(src, dest):
            print('but it\'s OK, the files were the same anyway')
        else:
            print('DISASTER - the files were different!\n\n\n')
    else:
        shutil.copy(src,dest)
        
def SafeMove(src, dest):
    if os.path.exists(dest):
        print('warning, tried to overwrite %s with %s'%(dest, src))
        if filecmp.cmp(src, dest):
            print('but it\'s OK, the files were the same anyway')
        else:
            print('DISASTER - the files were different!\n\n\n')
    else:
        shutil.move(src,dest)      
        
def ReverseDictionary(input_dict):
    return dict((v,k) for k,v in input_dict.items())

def GetFilename(filename_or_path):
    return str(filename_or_path).split('/')[-1]

def SafeCopyDir(src_dir, dest_dir, prepend_date=True, skip_eko_files=True):
    files = [_ for _ in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir,_))] #grab files, reject directories
    for fname in files:
        if skip_eko_files:
            if fname.find('eko')!=-1: continue
        source_filename = os.path.join(src_dir, fname)
        if prepend_date:
            try:
                d_file = pf.open(source_filename)
                primaryHeader = d_file[0].header
                date = primaryHeader['DATE-OBS'].split('T')[0]
                d_file.close()
            except Exception as e:
                print(repr(e))
                print('no date found for prepending in %s'%source_filename)
                date = ''
                
        dest_filename = os.path.join(dest_dir, date + fname)
        if os.path.exists(dest_filename):
            print('warning: tried to overwrite %s with %s'%(dest_filename, source_filename))
            if filecmp.cmp(source_filename, dest_filename):
                print('but it\'s OK, the files were the same anyway')
            else:
                print('\n\n *** DISASTER - the files were different!*** \n\n\n')
        else:
            shutil.copy(source_filename,dest_filename)

def ShowExp(exp, percentile=0.5, saveAs=None, dontShow=False):
    data = exp.getMaskedImage().getImage().getArray()
    plt.figure(figsize=(10,10))
    plt.imshow(data, cmap='gray', vmin=np.percentile(data, percentile), vmax=np.percentile(data, 100-percentile), origin='bl')
    if not dontShow:
        plt.show()
    if saveAs:
        plt.savefig(saveAs)

def BoxcarAverage1DArray(data, length):
    return np.convolve(data, np.ones((length,))/length,mode='valid')


def TranslatePImMSToTimepix(in_file, run_num, out_path):
    data = np.loadtxt(in_file)
    x = data[:,0] 
    y = data[:,1] 
    t = data[:,2]
    shot = data[:,3]-1 #pimms shot number is 1-based, we want to fix that
    
    n_shots = int(shot[-1])
    for i in range(n_shots):
        indices = np.where(shot == i)
        lines = []
        for index in indices[0]:
            lines.append(str(int(x[index])) + '\t'+ str(int(y[index])) + '\t'+ str(int(t[index]))+ '\n')
        
        file = open(out_path + str(run_num).rjust(2,'0') +'_'+ str(i+1).rjust(4,'0')+'.txt','w')
        file.writelines(lines)
        file.close()
    
    

def TimepixToExposure_binary(filename, xmin, xmax, ymin, ymax, mask_pixels=np.ones((1), dtype = np.float64)):
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
            my_array[y,x] = 1
        my_image = makeImageFromArray(my_array*mask_pixels.transpose())
        return_npix = (my_array*mask_pixels.transpose()).sum() #apply the mask, *then* sum!
    
    else:   
        x = data[:, 0] 
        y = data[:, 1] 
        t = data[:, 2]
        for pointnum in range(len(x)):
            if x[pointnum] >= xmin and x[pointnum] <= xmax and y[pointnum] >= ymin and y[pointnum] <= ymax:
                my_array[y[pointnum],x[pointnum]] = 1
        
        my_image = makeImageFromArray(my_array*mask_pixels.transpose())
        return_npix = (my_array*mask_pixels.transpose()).sum() #apply the mask, *then* sum!
        
    return my_image, return_npix



def MakeMaskArray(mask_list):
    import numpy as np
    mask_array = np.ones((256,256), dtype = np.int32)
    
    for i in range(len(mask_list[0])):
        y = mask_list[0][i]
        x = mask_list[1][i]
        mask_array[y][x] = 0
    return mask_array


def MaskBadPixels(data_array, mask_list):
    mask_array = MakeMaskArray(mask_list)
    data_array *= mask_array
    
    
def GeneratePixelMaskListFromFileset(path, noise_threshold = 0.03, xmin = 0, xmax = 255, ymin = 0, ymax = 255, file_limit = 1e6):
    import numpy as np
    import os
#     intensity_array = MakeCompositeImage_Timepix(path, 0, 255, 0, 255, 0, 9999, -99999, 99999, return_raw_array=True)
    intensity_array = MakeCompositeImage_Timepix(path, xmin, xmax, ymin, ymax, 0, file_limit, -99999, 99999, return_raw_array=True)
    nfiles = len(os.listdir(path))
    mask_pixels = np.where(intensity_array >= noise_threshold*(nfiles))

    return mask_pixels
    

def ViewMaskInDs9(mask_array):
    import lsst.afw.display.ds9 as ds9
    ds9.mtv(makeImageFromArray(mask_array))
    

def ViewIntensityArrayInDs9(intensity_array, savefile = None):
    import lsst.afw.display.ds9 as ds9
    ds9.mtv(makeImageFromArray(100*intensity_array/float(intensity_array.max())))
    if savefile is not None:
        arg = 'saveimage jpeg ' + str(savefile) + ' 100'
        ds9.ds9Cmd(arg)





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

def ReadTektronixWaveform(filename):
    import pylab as pl
    data = pl.loadtxt(filename, delimiter = ',', skiprows = 18, usecols = [3,4])
    xs = data[:,0]
    ys = data[:,1]
    return xs, ys
    
def ReadBNL_PMTWaveform(filename):
    import pylab as pl
    data = pl.loadtxt(filename, skiprows = 7)
#     pl.loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin)
    xs = data[:,0]
    ys = data[:,1]
    return xs, ys
    
def GetRawTimecodes_SingleFile(filename, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999,tmin=-9999999,tmax=9999999, offset_us = 0):
    import string
    
    timecodes = []
    datafile = open(filename)
    
    for line in datafile.readlines():
        x,y,timecode = string.split(str(line),'\t')
        x = int(x)
        y = int(y)
        timecode = int(timecode)
        if x >= winow_xmin and x <= winow_xmax and y >= winow_ymin and y <= winow_ymax: 
            if (timecode <= tmax) and (timecode >= tmin):
                timecodes.append(timecode) 

    return timecodes

def GetTimecodes_AllFilesInDir(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, translate_to_us = False, glitch_threshold = 20000, checkerboard_phase = None, noise_mask = None):
    import string, os
    
    timecodes = []
    files = []
    
    for filename in os.listdir(path):
        files.append(path + filename)
        

    nfiles = 0
    for filename in files:
        if str(filename).find('.DS')!=-1:continue
        
        datafile = open(filename)
        nfiles += 1
        if len(files)>500 and (nfiles % 500 == 0): print('Loaded %s of %s files'%(nfiles, len(files)))
        lines = datafile.readlines()
        
        if len(lines) > glitch_threshold: continue #skip files which glitched (most pixels hit - parameter may need tuning)
        for line in lines:
            x,y,timecode = string.split(str(line),'\t')
            x = int(x)
            y = int(y)
            
            if noise_mask is not None:
                if noise_mask[x][y] == 0: continue
            
            timecode = int(timecode)
            if timecode == 11810: continue  #discard overflows
            if timecode == 1: continue      #discard noise hits
            
            if checkerboard_phase is not None:
                if (x+y)%2 == checkerboard_phase:
                    timecode -= 1 
            
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                if translate_to_us == True:
                    actual_offset_us = 250 - offset_us
                    time_s = (11810. - timecode) * 20e-9
                    time_us = (time_s *1e6)- actual_offset_us
                    timecodes.append(time_us)
                else:
                    timecodes.append(timecode)

    print("Loaded data from %s files"%nfiles)
    return timecodes


def GetMaxClusterTimecodes_AllFilesInDir(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, checkerboard_phase = None, npix_min = 1):
    import os
    all_timecodes = []
    
    files = os.listdir(path)
    nfiles = len(files)
    for i,filename in enumerate(files):
        if i%500 == 0: print('Centroided %s of %s files'%(i,nfiles))
        codes = Clusterfind_max_timecode_one_file(path + filename, winow_xmin=winow_xmin, winow_xmax=winow_xmax, winow_ymin=winow_ymin, winow_ymax=winow_ymax, checkerboard_phase=checkerboard_phase, npix_min=npix_min)
        all_timecodes.extend(codes)
        
    return all_timecodes

def Clusterfind_max_timecode_one_file(filename, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, glitch_threshold = 20000, checkerboard_phase = None, npix_min = 1):
    if str(filename).find('.DS')!=-1:return
    
    import lsst.afw.detection as afwDetect
    import string, os
    
    cluster_sizes = []
    timecodes = []
    files = []
    
    thresholdValue = 1
    npixMin = npix_min
    grow = 0
    isotropic = False
    
    image = TimepixToExposure(filename, winow_xmin, winow_xmax, winow_ymin, winow_ymax)
    
    threshold = afwDetect.Threshold(thresholdValue)
    footPrintSet = afwDetect.FootprintSet(image, threshold, npixMin)
    footPrintSet = afwDetect.FootprintSet(footPrintSet, grow, isotropic)
    footPrints = footPrintSet.getFootprints()
    
    for footprintnum, footprint in enumerate(footPrints):
        npix = afwDetect.Footprint.getNpix(footprint)
        cluster_sizes.append(npix)
        
#         if npix >= 4:
        box = footprint.getBBox()
        bbox_xmin = box.getMinX()
        bbox_xmax = box.getMaxX() + 1
        bbox_ymin = box.getMinY()
        bbox_ymax = box.getMaxY() + 1
          
        data = image.getArray()[bbox_ymin:bbox_ymax,bbox_xmin:bbox_xmax]        
#         x,y,t,chisq = CentroidTimepixCluster(data, fit_function = 'gaus')
#         timecodes.append(t)

##         centroid_x, centroid_y = footprint.getCentroid()
##         x += bbox_xmin
##         y += bbox_ymin

        if checkerboard_phase is not None:
            print('WARNING - Not yet implemented')
            exit()
            if (bbox_xmin+bbox_ymin)%2 == checkerboard_phase:
                timecode -= 1 

        timecodes.append(GetMaxClusterTimecode(data))

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
        if (num % 10 == 0): print('loaded %s files'%num)
        
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
#                 print time_us
#                 exit()
                if time_us>=tmin_us and time_us<= tmax_us:
                    xs.append(x)
                    ys.append(y)
                    ts.append(time_us)
        
        if maxfiles != None and num == maxfiles:
            return xs, ys, ts

#     if return_as_ndarray:
#         return 

    return xs, ys, ts  



def ShowRawToF_whole_dir(path, invert = False, logy = False):
    import pylab as pl
    raw_codes = GetTimecodes_AllFilesInDir(path, 0, 256, 0, 256, 0, checkerboard_phase = None)

    fig = pl.figure(figsize=(14,10))
    
    if invert:
        n_codes, bins, patches = pl.hist([11810-i for i in raw_codes], bins = 11810, range = [0,11810])
    else:
        n_codes, bins, patches = pl.hist(raw_codes, bins = 11810, range = [0,11810])
    
    if logy: pl.yscale('log', nonposy='clip')

    
    pl.show()

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
        if (num % 10 == 0): print('loaded %s files'%num)
        
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


def MakeCompositeImage_Timepix(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, maxfiles = None, t_min = -9999, t_max = 9999, return_raw_array = False):
    from lsst.afw.image import makeImageFromArray
    import string, os
    import numpy as np
    import pylab as pl
    
    my_array = np.zeros((256,256), dtype = np.int32)

    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    for filenum, filename in enumerate(files):
        if filenum % 500 ==0: print("Compiled %s files"%filenum)
        
        xs, ys, ts = GetXYTarray_SingleFile(filename, winow_xmin, winow_xmax, winow_ymin, winow_ymax)
#         if len(xs) > 5000: continue # skip glitch files
        
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            t = ts[i]
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                if t>=t_min and t<=t_max:
                    my_array[x,y] += 1
      
      
        if maxfiles != None and filenum >= maxfiles:
            if return_raw_array: return my_array
            my_image = makeImageFromArray(my_array)
            return my_image
        
    my_image = makeImageFromArray(my_array)
    if return_raw_array: return my_array
    return my_image



def MakeCompositeImage_PImMS(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, maxfiles = None, t_min = -9999, t_max = 9999, return_raw_array = False):
    from lsst.afw.image import makeImageFromArray
    import string, os
    import numpy as np
    import pylab as pl
    
    my_array = np.zeros((72,72), dtype = np.int32)

    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    for filenum, filename in enumerate(files):
        if filenum % 500 ==0: print("Compiled %s files"%filenum)
        
        xs, ys, ts = GetXYTarray_SingleFile(filename, winow_xmin, winow_xmax, winow_ymin, winow_ymax)
#         if len(xs) > 5000: continue # skip glitch files
        
        for i in range(len(xs)):
            x = xs[i]
            y = ys[i]
            t = ts[i]
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                if t>=t_min and t<=t_max:
                    my_array[x,y] += 1
      
      
        if maxfiles != None and filenum >= maxfiles:
            if return_raw_array: return my_array
            my_image = makeImageFromArray(my_array)
            return my_image
        
    my_image = makeImageFromArray(my_array)
    if return_raw_array: return my_array
    return my_image



def TimecodeTo_us(timecode):
    return (11810. - timecode) * 0.02 # 20e-9 * 1e6


def OpenTimepixInDS9(filename, binary = False):
    import lsst.afw.display.ds9 as ds9
    if binary:
        image,dummy = TimepixToExposure_binary(filename, 0,255,0,255)
    else:
        image = TimepixToExposure(filename, 0,255,0,255)

    try:
        ds9.initDS9(False)
    except ds9.Ds9Error:
        print('DS9 launch bug error thrown away (probably)')

    ds9.mtv(image)
    
    
def OpenImageInDS9(image):
    import lsst.afw.display.ds9 as ds9

    try:
        ds9.initDS9(False)
    except ds9.Ds9Error:
        print('DS9 launch bug error thrown away (probably)')

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
        if (filenum % 100 == 0): print('loaded %s files'%filenum)
        
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
      
def MakeTimeSlices(inpath, slicelist, outpath):
    from lsst.afw.image import makeImageFromArray
    import string, os
    import numpy as np
    from TrackViewer import TrackToFile_ROOT_2D_3D
    
    image_array = []
    for i in slicelist:
       image_array.append(np.zeros((256,256), dtype = np.int32)) 
    
    xs, ys, ts = GetXYTarray_AllFilesInDir_Raw_Timecodes(inpath)
    
    for slicenum,slice in enumerate(slicelist):
        t_min = slice[0]
        try:
            t_max = slice[1]
        except:
            t_max = t_min
            
        try: #NB try/except blocks do need to be separate here
            prefix = slice[2]
        except:
            prefix = ''
            

            
        for i in range(len(xs)):
            t = ts[i]
            if t>=t_min and t<=t_max:
                image_array[slicenum][xs[i],ys[i]] += 1
      
      
        for i in range(1,3):
            if t_min == t_max:
                outname = outpath + str(t_min) + '_boxcar_' + str(i) + '.png'
            else:
                outname = outpath + prefix + 'range_' + str(t_min) + '_' + str(t_max) + '_boxcar_' + str(i) + '.png'
        
            avergaged_array = BoxcarAverage2DArray(image_array[slicenum], i)
            TrackToFile_ROOT_2D_3D(avergaged_array, outname, plot_opt='surf1', zmax_supress_ratio = 0.6, log_z = False, force_aspect= True, fitline = None)
        
#         TrackToFile_ROOT_2D_3D(image_array[slicenum], outname, plot_opt='surf1', zmax_supress_ratio = 0.5, log_z = False, force_aspect= True, fitline = None)
    
        
    return


def BoxcarAverage2DArray(array, boxcar_size):
    import numpy as np
    xsize, ysize = array.shape
    if boxcar_size == 1:
        return array
    if boxcar_size < 1:
        print("Error - Boxcar size cannot be less than 1")
        exit()
        
    ret = np.zeros((xsize - (boxcar_size - 1),ysize - (boxcar_size - 1)), dtype = np.float32)
    for x in range(xsize - boxcar_size + 1):
        for y in range(ysize - boxcar_size + 1):
            av = np.average(array[x:x+boxcar_size,y:y+boxcar_size])
            ret[x,y] = av  
    return ret 

def GetXYTarray_AllFilesInDir_Raw_Timecodes(path, winow_xmin = 0, winow_xmax = 999, winow_ymin = 0, winow_ymax = 999, offset_us = 0, maxfiles = None):
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
        if (num % 100 == 0): print('loaded %s files'%num)
        
        #handle problem with the way loadtxt reads single line data files
        if data.shape == (3,): 
            x = int(data[0])
            y = int(data[1])
            timecode = int(data[2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                xs.append(x)
                ys.append(y)
                ts.append(timecode)
            continue
        
        #extract data for multiline files
        if len(data) > 10000: continue #skip glitch files
        for i in range(len(data)):
            x = int(data[i,0])
            y = int(data[i,1])
            timecode = int(data[i,2])
            if x>=winow_xmin and x<=winow_xmax and y>=winow_ymin and y<=winow_ymax:
                xs.append(x)
                ys.append(y)
                ts.append(timecode)
        
        if maxfiles != None and num == maxfiles:
            return xs, ys, ts

    return xs, ys, ts  



def MakeToFSpectrum(input_path, save_path, xmin=0, xmax=255, ymin=0, ymax=255, translate_to_us = True, time_zoom_region = [0,11810]):
    import pylab as pl
    
    timecodes = GetTimecodes_AllFilesInDir(input_path, xmin, xmax, ymin, ymax, 110, translate_to_us)
    print('Total number of timecodes read in = %s' %len(timecodes))
     
    if translate_to_us:
        tmin = 5
        tmax = 25
        bins = (((tmax-tmin))*50) - 1 #1 timecode per bin
    else:
        tmin = 0         # full range
        tmax = 11810     # full range
        bins = (tmax-tmin) +1 
     
    fig = pl.figure()
    
    vals, bins, pathches = pl.hist(timecodes, bins = bins, range = [tmin,tmax]) #make the histogram of the timecodes
    
#     pl.ylim([0,2000]) #for clipping the y-axis
    
    if translate_to_us:
        pl.xlabel('ToF (us)', horizontalalignment = 'right' )
    else:
        pl.xlabel('Timecodes', horizontalalignment = 'right' )
    
    pl.title('Timepix ToF Spectrum')
    fig.savefig(save_path + '_ToF_Full.png')

    tmin, tmax = time_zoom_region
    ylim = max(vals[tmin:tmax])
    ylim *= 1.2
    pl.ylim([0,ylim]) #for clipping the x-axis
    pl.xlim([tmin,tmax]) #for clipping the x-axis
    fig.savefig(save_path + '_ToF_ROI.png')
    print('Finished making ToF')
    
    
def CentroidTimepixCluster(data, save_path = None, fit_function = None):
    import numpy as np
    # from ROOT import *
    from ROOT import TH2F, TCanvas, TBrowser, TF2
    import ROOT
    gROOT.SetBatch(1) #don't show drawing on the screen along the way    
    
    nbinsx = xmax = max(data.shape[0], data.shape[1])
    nbinsy = ymax = max(data.shape[0], data.shape[1])
    
    xlow = 0
    ylow = 0
    
    tmin = np.amin(data[np.where(data >= 1)])
    
    if save_path!= None: 
        c1 = TCanvas( 'canvas', 'canvas', 1200,1000) #create canvas
    
    image_hist = TH2F('', '',nbinsx,xlow,xmax,nbinsy, ylow, ymax)
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                image_hist.Fill(float(x),float(y),float(value-tmin))
       
       
    if fit_function == 'gaus':
        fit_func = TF2("f2","[0]*TMath::Gaus(x,[1],[2])*TMath::Gaus(y,[3],[4])",0,xmax,0,ymax)
        fit_func.SetParameters(10,3,3,3,3)
    elif fit_function == 'p2':
        fit_func = TF2("f2",'[0]*(x-[1])^2 + [2]*(y-[3])^2 + [4]',0,xmax, 0, ymax)
        fit_func.SetParameters(10,3,3,3,3)
    elif fit_function == 'p4':
        fit_func = TF2("f2",'[0]*(x-[1])^4 + [2]*(y-[3])^4 + [4]',0,xmax, 0, ymax)
        fit_func.SetParameters(-0.002,10,-0.002,8,15)
    elif fit_function == None:
       print('Warning - not fitting clusters')
    else:
        print('Error - unknown fit function')
        exit()
  
    if fit_function != None:
        image_hist.Fit(fit_func, 'MEQ')
        true_xmax = fit_func.GetParameter(1)
        true_ymax = fit_func.GetParameter(3)
        true_tmax = fit_func.Eval(true_xmax, true_ymax) + tmin
        chisq =  fit_func.GetChisquare()
        NDF = fit_func.GetNDF()
        try:
            chisqred = chisq/NDF
        except:
            chisqred = 999999

    
    if save_path!= None:
        if fit_function == None:
            image_hist.Draw('lego20z')
            image_hist.GetZaxis().SetRangeUser(0,np.ceil(data.max() - tmin))
            image_hist.SetStats(False)
            c1.SaveAs(save_path)
        else:
            fit_func.SetNpx(1000)
            image_hist.Draw('lego20')
            fit_func.Draw("same")
            zrange = np.ceil(true_tmax - tmin)
            zrange = max(zrange,10, np.ceil(data.max() - tmin))
            image_hist.GetZaxis().SetRangeUser(0,zrange)
            image_hist.SetStats(False)
            c1.SaveAs(save_path)
            del c1
    
    if fit_function != None: return true_xmax, true_ymax, true_tmax, chisqred
    return 0, 0, 0, 0
    
def GetMaxClusterTimecode(data):
    return np.amax(data[np.where(data >= 1)])

# def GetMinClusterTimecode(data):
#     return np.amin(data[np.where(data >= 1)])
    
def GetAllTimecodesInCluster(data):
    timecodes = []
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            value = data[x,y]
            if value != 0:
                timecodes.append(value)
    
    return timecodes
    
def Combine_and_pickle_dir(path, output_pickle):
    import pickle as pickle
    import os
    import pylab as pl
    
    files = []
    for filename in os.listdir(path):
        files.append(path + filename)

    xs, ys, ts = [], [], []

    for filenum, filename in enumerate(files):
        data = pl.loadtxt(filename, usecols = (0,1,2))
        if (filenum % 500 == 0): print('loaded %s of %s files'%(filenum, len(files)))
        
        #handle problem with the way loadtxt reads single line data files
        if data.shape == (3,): 
            xs.append(int(data[0]))
            ys.append(int(data[1]))
            ts.append(int(data[2]))
        
        #extract data for multiline files
        if len(data) > 10000: continue #skip glitch files
        for i in range(len(data)):
            xs.append(int(data[i,0]))
            ys.append(int(data[i,1]))
            ts.append(int(data[i,2]))
            
    x_array = np.asarray(xs, dtype = np.int16)
    y_array = np.asarray(ys, dtype = np.int16)
    t_array = np.asarray(ts, dtype = np.int16)

    data_array = np.ndarray([len(x_array),3], dtype = np.int16)
    data_array[:,0] = x_array
    data_array[:,1] = y_array
    data_array[:,2] = t_array
    
    pickle.dump(data_array, open(output_pickle,'wb'), pickle.HIGHEST_PROTOCOL)
    
def Load_XYT_pickle(filename):
    import pickle as pickle
    return pickle.load(open(filename, 'rb'))
    
def XYT_to_image(xyt_array, display = False):
    import numpy as np
    from lsst.afw.image import makeImageFromArray
    if display:
        import lsst.afw.display.ds9 as ds9
        try:
            ds9.initDS9(False)
        except ds9.Ds9Error:
            print()

    my_array = np.zeros((256,256), dtype = np.int32)
     
    xs = xyt_array[:, 0] 
    ys = xyt_array[:, 1] 
        
#     for x,y in zip(xs,ys):
#         my_array[y,x] += 1
         
    n_counts = 0
    for x,y in zip(xs,ys):
        if n_counts >= 1000000: break
        n_counts += 1
        my_array[y,x] += 1
         
    my_image = makeImageFromArray(my_array)
    if display: ds9.mtv(my_image)
     
    return my_image
    
    
def Make3DScatter(xs, ys, ts, tmin = -9999999, tmax = 9999999, xminmax = [0,255], yminmax = [0,255], savefile = ''):
    import pylab as pl
    import numpy as np
    
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    w = np.where(ts >= tmin and ts <= tmax)
    ax.scatter(xs[w], ys[w], ts[w])
    pl.xlim(xminmax)
    pl.ylim(yminmax)
    if savefile != '':
        fig.savefig(savefile)
    else:
        pl.show()
