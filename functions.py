def raDecStrToFloat(raStr, decStr):
    h, m, s = [float(_) for _ in raStr.split(":")]
    ra = 15*(h + (m + s/60.0)/60.0)

    if decStr[0] == '-':
        decStr = decStr[1:]
        sign = -1
    else:
        sign = +1
    d, m, s = [float(_) for _ in decStr.split(":")]
    dec = sign*(d + (m + s/60.0)/60.0)

    return ra, dec

def SideBySide(images, half_stretch=3, smoothing=1, fix_scales=True, saveas = '', colormap='gray', logscale=False):
    import matplotlib.pyplot as plt
    import numpy as np

    if type(images)!=list:
        print 'Please provide a list of images to be plotted side by side'
        return
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage.filters import gaussian_filter

    n_images= len(images)
    fig = plt.figure(figsize = [8*n_images,15])
    
    
    for i,image in enumerate(images):
        if str(type(images[i]))=='<class \'lsst.afw.image.imageLib.ExposureF\'>':
            raw_data = image.getMaskedImage().getImage().getArray()
        elif str(type(images[i]))=='<type \'numpy.ndarray\'>':
            raw_data = images[i]
        else:
            print 'Error: You passed a list of type %s,\
            you can either pass a list of arrays or a list\
            of ExposureF\'s (at the moment)'%type(images[i])
            return
        
        im_data = gaussian_filter(raw_data,smoothing)
        
        if logscale:
            im_data[im_data <= 0] = 1e-9
            im_data = np.log10(im_data)
        
        im_mean = np.mean(im_data)

        if i==0:
            if half_stretch == -1 and logscale==False:
                vmin = np.min(im_data)
                vmax = np.max(im_data)
            elif half_stretch == -1 and logscale==True:
                vmin = 0
                vmax = np.max(im_data)    
            elif half_stretch == 'auto'and logscale==False:
                vmin = np.percentile(im_data, 1)
                vmax = np.percentile(im_data, 98)
            elif half_stretch == 'auto'and logscale==True:
                vmin = 0
                vmax = np.percentile(im_data, 98)
            else:
                vmin = np.median(im_mean) - half_stretch
                vmax = np.median(im_mean) + half_stretch
        elif fix_scales==False:
            if half_stretch == -1 and logscale==False:
                vmin = np.min(im_data)
                vmax = np.max(im_data)
            elif half_stretch == -1 and logscale==True:
                vmin = 0
                vmax = np.max(im_data)    
            elif half_stretch == 'auto'and logscale==False:
                vmin = np.percentile(im_data, 1)
                vmax = np.percentile(im_data, 98)
            elif half_stretch == 'auto'and logscale==True:
                vmin = 0
                vmax = np.percentile(im_data, 98)
            else:
                vmin = np.median(im_mean) - half_stretch
                vmax = np.median(im_mean) + half_stretch
#        if i==0:
#            vmin = im_mean - half_stretch
#            vmax = im_mean + half_stretch
#        elif fix_scales==False:
#            vmin = im_mean - half_stretch
#            vmax = im_mean + half_stretch

        axes = fig.add_subplot(1,n_images,i+1)
        im = axes.imshow(im_data, interpolation = 'none', cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
    
    if saveas != '':
        try:
            fig.savefig(saveas)
            from os import getcwd
            if saveas[0]=='/':
                print "Saved figure to %s"%saveas
            else:
                print "Saved figure to %s"%(str(getcwd() + '/' + saveas))
        except:
            print "failed to save figure as %s"%saveas


def getClippedMeanandStddev(data, nsig=3):
    from scipy.stats import sigmaclip
    import numpy as np
    clipped_array = sigmaclip(data, low=nsig, high=nsig)[0]
    return np.mean(clipped_array), np.std(clipped_array)


def focal_plane_to_image(im_data, save_file='', half_stretch=3, smoothing=0, colormap='gray'):
    if save_file == '':
        print 'Must supply output filename'
        return
    
#     from scipy.ndimage.filters import gaussian_filter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig = plt.figure(figsize = [20,20])
    
#     im_data = gaussian_filter(image.getMaskedImage().getImage().getArray(),smoothing)
#     im_data = image.getMaskedImage().getImage().getArray()
    print im_data.shape
    im_mean = np.mean(im_data)
    vmin = im_mean - half_stretch
    vmax = im_mean + half_stretch

    axes = fig.add_subplot(1,1,1)
    im = axes.imshow(im_data, interpolation = 'none', cmap=colormap, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    
#     try:
    fig.savefig(save_file)
    from os import getcwd
    if save_file[0]=='/':
        print "Saved figure to %s"%save_file
    else:
        print "Saved figure to %s"%(str(getcwd() + '/' + save_file))
#     except:
#         print "failed to save figure as %s"%save_file

def ShowSpot(raw_data, half_stretch=-1, smoothing=0, saveas = '', colormap='CMRmap', title='', logscale=False, useClippedMean=False, n_sig_clip=3):
    '''set half_stretch to -1 for full range, to'auto' for using percentiles, otherwise a numerical value from the mean'''
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.ndimage.filters import gaussian_filter

    fig = plt.figure(figsize=(10,10))
    if title != '': plt.title(str(title), fontsize=25)

    im_data = gaussian_filter(raw_data,smoothing)
    if useClippedMean:
        im_mean, dummy = getClippedMeanandStddev(im_data, nsig=n_sig_clip)
    else:
        im_mean = np.mean(im_data)

    if logscale:
        im_data[im_data <= 0] = 1e-9
        im_data = np.log10(im_data)
    
    if half_stretch == -1 and logscale==False:
        vmin = np.min(im_data)
        vmax = np.max(im_data)
    elif half_stretch == -1 and logscale==True:
        vmin = 0
        vmax = np.max(im_data)    
    elif half_stretch == 'auto'and logscale==False:
        vmin = np.percentile(im_data, 1)
        vmax = np.percentile(im_data, 98)
    elif half_stretch == 'auto'and logscale==True:
        vmin = 0
        vmax = np.percentile(im_data, 98)
    else:
        if useClippedMean:
            im_mean, dummy = getClippedMeanandStddev(im_data, nsig=n_sig_clip)
            vmin = im_mean - half_stretch
            vmax = im_mean + half_stretch            
        else:
            vmin = np.median(im_mean) - half_stretch
            vmax = np.median(im_mean) + half_stretch

    axes = fig.add_subplot(1,1,1)
    
    im = axes.imshow(im_data, interpolation = 'nearest', cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    if logscale: cbar.ax.set_yticklabels([np.round(10.**float(_.get_text()),0) for _ in cbar.ax.get_yticklabels()])
            
    plt.tight_layout()
    
    if saveas != '':
        try:
            fig.savefig(saveas)
            from os import getcwd
            if saveas[0]=='/':
                print "Saved figure to %s"%saveas
            else:
                print "Saved figure to %s"%(str(getcwd() + '/' + saveas))
        except:
            print "failed to save figure as %s"%saveas

            
def SurfPlot(data):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cbook
    from matplotlib import cm
    from matplotlib.colors import LightSource
    import matplotlib.pyplot as plt
    import numpy as np

    ncols, nrows = data.shape
    if ncols * nrows > 20000: print 'Warning, surface plotting things of this size is slow...'
    z = data
    x = np.linspace(0, ncols, ncols)
    y = np.linspace(0, nrows, nrows)
    x, y = np.meshgrid(x, y)
    
    region = np.s_[0:min(ncols,nrows), 0:min(ncols,nrows)]
    x, y, z = x[region], y[region], z[region]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(10,10))

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.CMRmap, vert_exag=0.01, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=True, shade=False)
    plt.show()

    
def SumSourcePixels(source, parent_image):
    import lsst.afw.detection as afwDetect
    import numpy as np

    fp = source.getFootprint()
    array = np.zeros(fp.getArea(), dtype=parent_image.getArray().dtype)
    afwDetect.flattenArray(fp, parent_image.getArray(), array, parent_image.getXY0())
    flux = array.sum(dtype=parent_image.getArray().dtype)
    return flux

def SumFootprintPixels(footprint, parent_image):
    import lsst.afw.detection as afwDetect
    import numpy as np

    array = np.zeros(footprint.getArea(), dtype=parent_image.getArray().dtype)
    afwDetect.flattenArray(footprint, parent_image.getArray(), array, parent_image.getXY0())
    flux = array.sum(dtype=parent_image.getArray().dtype)
    return flux
   
def printHeaderData(filename,hduNum=0, max_lines = 1e9, return_not_print=False):
    import pyfits as pf
    if return_not_print: ret = []
        
    this_file = pf.open(filename)
    primaryHDU = this_file[hduNum]
    for i, key in enumerate(primaryHDU.header):
        if i >= max_lines: break
        if return_not_print:
            ret.append(str(key) + ':' + str(primaryHDU.header[key]))
        else:
            print key, primaryHDU.header[key]
    
    this_file.close()
    if return_not_print: return ret
    
def CoaddExposures(exposures, outnumber, datapath='/nfs/lsst2/photocalData/data/observer2/', verbose=False, normalise=False):
    import pyfits as pf
    import shutil
    import sys
    
    N_HDUS = 70
    print 'Coadding %s'%(exposures); sys.stdout.flush()
    n_exp = float(len(exposures))

    filenames = [datapath + 'DECam_00' + str(_) + '.fits.fz' for _ in exposures]
    outfilename = datapath + 'DECam_0' + str(9000000 + outnumber) + '.fits.fz' 
    
    shutil.copyfile(filenames[0],outfilename,)
    out_file = pf.open(outfilename, mode='update')
    
    primaryHeader = out_file[0].header
    total_EXPTIME  = primaryHeader['EXPTIME']
    total_EXPREQ   = primaryHeader['EXPREQ']
    total_DARKTIME = primaryHeader['DARKTIME']

    # convert all arrays to floats for summing and dividing purposes
    if verbose: print 'loading first file & converting dtype'
    for hdu in range(1, N_HDUS+1):
        out_file[hdu].data = out_file[hdu].data.astype(np.float32)
    
    # add other files to the original, collecting relevant metadata
    for i, filename in enumerate(filenames[1:]):
        this_file = pf.open(filename)
        total_EXPTIME  += this_file[0].header['EXPTIME']
        total_EXPREQ   += this_file[0].header['EXPREQ']
        total_DARKTIME += this_file[0].header['DARKTIME']

        for hdu in range(1, N_HDUS+1):
            if verbose: print 'adding hdu %s for file %s of %s'%(hdu,i+2,n_exp)
            out_file[hdu].data += this_file[hdu].data
    
    # Normalise
    if normalise:
        for hdu in range(1, N_HDUS+1):
            if verbose: print 'Normalising hdu %s'%hdu
            out_file[hdu].data /= n_exp

    # Update headers
    primaryHeader['nCOADDED'] = n_exp
    primaryHeader['filename'] = 'DECam_0' + str(9000000 + outnumber) + '.fits'
    primaryHeader['expnum']   = 9000000 + outnumber
    primaryHeader['COADD_OF'] = str(['DECam_00' + str(_) for _ in exposures]).translate(None, ''.join(['[',']',' ','\'']))
    primaryHeader['COADNUMS'] = (str(exposures).translate(None, ''.join(['[',']',' '])))
    if not normalise: n_exp = 1.
    primaryHeader['NORMED']   = str(normalise)
    primaryHeader['EXP_TOT']  = total_EXPTIME # always equal to the total exposure time
    primaryHeader['DARK_TOT'] = total_DARKTIME # always equal to the total darktime
    primaryHeader['EXP_T_EQ'] = total_EXPTIME / n_exp #equivalent expousre time, depending on noralisation
    primaryHeader['EXPREQ']   = total_EXPREQ / n_exp #equivalent EXPREQ time, depending on noralisation
    primaryHeader['DARKTIME'] = total_DARKTIME / n_exp #equivalent DARKTIME time, depending on noralisation

    if verbose: print 'Headers updated, writing to disk...'; sys.stdout.flush()
    out_file.flush()
    out_file.close()
    if verbose: print 'Fished coaddition of %s, written to %s'%(exposures, outfilename)

def GetExpNumsFromDir(path):
    import pyfits as pf
    import os,sys
    
    filenames = os.listdir(path)
    print 'Found %s files in %s\nOpening...'%(len(filenames), path); sys.stdout.flush()

    expNums = []
    for filename in filenames:
        this_file = pf.open(os.path.join(path,filename))
        expNums.append(this_file[0].header['EXPNUM'])
        this_file.close()

    expNums.sort()
    caret_sep = str(expNums).replace(', ','^').replace('[','').replace(']','')
    print caret_sep
    return expNums, caret_sep

def GetExpNumsPerFilterDictFromDir(path):
    import pyfits as pf
    import os, sys
    
    ret = {}
    filenames = os.listdir(path)
    print 'Found %s files in %s\nOpening...'%(len(filenames), path); sys.stdout.flush()

    for i, filename in enumerate(filenames):
        if i%100==0: print 'Processed %s of %s files...'%(i, len(filenames)); sys.stdout.flush()
        this_file = pf.open(os.path.join(path,filename))
        expNum = this_file[0].header['EXPNUM']
        filt = this_file[0].header['FILTER']
        this_file.close()
        
        if not filt in ret.keys():
            ret[filt] = []
            ret[filt].append(expNum)
        else:
            ret[filt].append(expNum)
    
    for filt, expNums in ret.iteritems():
        caret_sep = str(sorted(expNums)).replace(', ','^').replace('[','').replace(']','')
        print filt +': '+ caret_sep
    return ret

def PrintListCaretSeparated(input_list, sort=True):
    if sort: input_list.sort()
    # I should reallly learn how to use regex :/
    caret_sep = str(input_list).replace(', ','^').replace('[','').replace(']','').replace('\'','')
    print caret_sep
    return caret_sep

                
def ThresholdList(input_list, threshold, default_value=0, return_as_array=False):
    if return_as_array: return np.asarray([val if val>=threshold else default_value for val in input_list])
    return [val if val>=threshold else default_value for val in input_list]


def GetRerunAndRepoNames(path):
    '''Assumes a lot, but is useful if your path goes /some/path/to/repo_name/rerun/renum_name'''
    try:
        parts = path.split('rerun')
        rerun = parts[1].replace('/','')
        repo_path = parts[0]
        a = parts[100]
        repo_name = parts[0].split('/')[-2]
        return repo_path, repo_name, rerun
    except:
        print 'Failed to parse repo path \nIt likely did not conform to\n/some/path/to/repo_name/rerun/renum_name'
        return '', '', ''
    
def indexOfMax(data):
    '''For 1D data, returns the index of first point in the array that is equal to the maximum value'''
    import numpy as np
    return np.where(data==max(data))[0][0]
def indexOfMin(data):
    '''For 1D data, returns the index of first point in the array that is equal to the minimum value'''
    import numpy as np
    return np.where(data==min(data))[0][0]

# def remove_non_ascii(text):
# #     return ''.join([i if ord(i) < 128 else ' ' for i in text])
#     return ''.join(i for i in text if ord(i)<128)