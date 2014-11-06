import re
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
from lsst import afw
from lsst.pex.config import Config, Field, ConfigField, makePolicy
from lsst.meas.algorithms.detection import (SourceDetectionTask, estimateBackground, BackgroundConfig)
from IPython.parallel.controller.scheduler import numpy


class MuonConfig(Config):
    # TODO: strip this down to remove bloat
#    detection = ConfigField(dtype=SourceDetectionTask.ConfigClass, doc="Detection config")
    background = ConfigField(dtype=BackgroundConfig, doc="Background subtraction config")
#    cosmicray = ConfigField(dtype=measAlg.FindCosmicRaysConfig, doc="Cosmic-ray config")
#    psfSigma = Field(dtype=float, default=2.0, doc="PSF Gaussian sigma")
#    psfSize = Field(dtype=int, default=21, doc="PSF size (pixels)")

#    def setDefaults(self):
#        super(MuonConfig, self).setDefaults()
#        self.cosmicray.keepCRs = True # We like CRs!
#        self.cosmicray.nCrPixelMax = 1000000
#        self.cosmicray.minSigma = 5.0
#        self.cosmicray.min_DN = 1.0
#        self.cosmicray.cond3_fac2 = 0.4

def AssembleImage_bad_way(filename, overscan_x = 0, overscan_y = 0):
    from lsst.afw.geom.geomLib import Point2I
    from lsst.afw.geom.geomLib import Box2I
    import time

    imagelist = [afwImage.DecoratedImageI(filename, i + 2) for i in range(16)]
    
    panelsize_x = imagelist[1].getWidth()
    panelsize_y = imagelist[1].getHeight()
    
    xsize = 8 * panelsize_x - (8 * overscan_x)
    ysize = 2 * panelsize_y - (2 * overscan_y)
    outimage = afwImage.ImageI(Box2I(Point2I(0,0),Point2I(xsize,ysize)))

    ypanels = 2
    xpanels = 8
    
    t0 = time.time()
   
    for panely in range(ypanels):
        for panelx in range(xpanels):
            panelnum = (panely*xpanels) + panelx
            for pixely in range(panelsize_y):
                for pixelx in range(panelsize_x):
                    mainpixelx = pixelx + (panelx * panelsize_x)
                    mainpixely = pixely + (panely * panelsize_y)
                    outimage.set(mainpixelx,mainpixely,imagelist[panelnum].getImage().get(pixelx,pixely))
                                    
    t1 = time.time()
    print "image assembled in %.2f seconds" % (t1 - t0)      
    return outimage  









def _read_with_bg_subtraction(filename):
    config = MuonConfig()
    imagearray = []
    for i in range(2,18):
        original_image = afwImage.DecoratedImageF(filename, i)
        maskedImg = afwImage.MaskedImageF(original_image.getImage())
        exposure = afwImage.ExposureF(maskedImg)
        bg, bgSubExp = estimateBackground(exposure, config.background, subtract=True)
        bgSubImg = bgSubExp.getMaskedImage().getImage()
        imagearray.append(bgSubImg)
    return imagearray


def _read(filename):
    imagearray = [afwImage.DecoratedImageF(filename, i + 2) for i in range(16)]
    return imagearray


def _regionToBox(region):
    """Convert [x0:x1,y0:y1] to Box2I"""
    m = re.search(r"\[(\d+):(\d+),(\d+):(\d+)\]", region)
    assert m
    x0, x1, y0, y1 = map(int, m.groups())
    return afwGeom.Box2I(afwGeom.Point2I(x0 - 1, y0 - 1), afwGeom.Point2I(x1 - 1, y1 - 1)) # Fortran --> C


def _assemble_alt_metadata(filename, metadata_filename, subtract_background, gains, ADC_Offsets = None):
    if subtract_background and (ADC_Offsets != None):
        print "**** - Error - both DM and bias background subtraction selected - ****"
        print "Defaulting to only using DM subtraction"
        ADC_Offsets = None
        
    if subtract_background:
        print "Using DM background subtraction"
        imageList = _read_with_bg_subtraction(filename)
    else:
        imageList = _read(filename)
        
    if ADC_Offsets == None:
        ADC_Offsets = numpy.zeros(16, dtype = 'f8')
        
    if gains != None: # if statement just deals with the fact that _read_with_bg_subtraction doesn't return decorated images...
        if not subtract_background: # for normal, decorated imaged
            for i in range(len(imageList)):
                imageList[i].getImage().getArray()[:] = (imageList[i].getImage().getArray()[:] - ADC_Offsets[i] ) * gains[i]
        else: #for un-decorated images, as returned by _read (i.e. without background subtraction
            for i in range(len(imageList)):
                imageList[i].getArray()[:] = (imageList[i].getArray()[:] - ADC_Offsets[i] ) * gains[i]
        
    metadata_imageList = _read(metadata_filename)

    detsize = set([md_im.getMetadata().get("DETSIZE") for md_im in metadata_imageList])
    assert len(detsize) == 1, "Multiple DETSIZEs detected"
    detsize = detsize.pop()
    image = afwImage.ImageF(_regionToBox(detsize))

    for i in range(len(imageList)):
        header = metadata_imageList[i].getMetadata()
        ltm11, ltm22 = map(int, [header.get("LTM1_1"), header.get("LTM2_2")])
        assert abs(ltm11) == 1 and abs(ltm22) == 1, "Binned data detected"
        assert "LTM1_2" not in header.names() and "LTM2_1" not in header.names(), "Rotated data detected"
        datasec = _regionToBox(header.get("DATASEC"))
        detsec = _regionToBox(header.get("DETSEC"))
        if subtract_background:
            data = afwImage.ImageF(imageList[i], datasec)
        else:
            data = afwImage.ImageF(imageList[i].getImage(), datasec)
        data = afwMath.flipImage(data, ltm11 < 0, ltm22 < 0)
        target = image.Factory(image, detsec)
        target <<= data
    return image



def _assemble(filename, subtract_background, gains):
    if subtract_background:
        imageList = _read_with_bg_subtraction(filename)
    else:
        imageList = _read(filename)
        
    if gains != None:
        for i in range(len(imageList)):
            imageList[i].getImage().getArray()[:] *= gains[i]
 
    detsize = set([im.getMetadata().get("DETSIZE") for im in imageList])
    assert len(detsize) == 1, "Multiple DETSIZEs detected"
    detsize = detsize.pop()
    image = afwImage.ImageF(_regionToBox(detsize))
    for im in imageList:
        header = im.getMetadata()
        ltm11, ltm22 = map(int, [header.get("LTM1_1"), header.get("LTM2_2")])
        assert abs(ltm11) == 1 and abs(ltm22) == 1, "Binned data detected"
        assert "LTM1_2" not in header.names() and "LTM2_1" not in header.names(), "Rotated data detected"
        datasec = _regionToBox(header.get("DATASEC"))
        detsec = _regionToBox(header.get("DETSEC"))
        if subtract_background:
            data = afwImage.ImageF(im, datasec)
        else:
            data = afwImage.ImageF(im.getImage(), datasec)
        data = afwMath.flipImage(data, ltm11 < 0, ltm22 < 0)
        target = image.Factory(image, detsec)
        target <<= data
    return image


def AssembleImage(filename, metadata_filename = None, subtract_background = False, gain_correction_list = None, ADC_Offsets = None):
    if gain_correction_list != None: assert len(gain_correction_list) == 16
    if metadata_filename is not None:
        image = _assemble_alt_metadata(filename, metadata_filename, subtract_background, gain_correction_list, ADC_Offsets)
    else:
        image = _assemble(filename, subtract_background, gain_correction_list)
    
    return image

def MakeBiasImage(bias_files, metadata_filename, gain_correction_list = None):
    '''Make a naive composite bias image, i.e. not rejecting frames with cosmics in'''
    from os import listdir
    from os.path import expanduser

    home_dir = expanduser("~")
    bias_image = AssembleImage(bias_files[0], metadata_filename)
    bias_image -= bias_image

    if gain_correction_list != None:
        for i in range(len(gain_correction_list)):
            gain_correction_list[i] = 1. / gain_correction_list[i]

    i = 0
    for filename in bias_files:
        bias_image += AssembleImage(filename, metadata_filename, gain_correction_list)
        i += 1
        
    bias_image /= float(i)
    return bias_image

def MakeBiasImage_SingleAmp(path, amp_number_zero_based):
    '''Make a naive composite bias image, i.e. not rejecting frames with cosmics in'''
    from os import listdir

    bias_files = []
    for filename in listdir(path):
        if filename.find('bias') != -1:
            bias_files.append(path + filename)

    # easiest way to make an image of the right size, don't judge me :/
    bias_image = afwImage.ImageF(bias_files[0])
    bias_image -= bias_image

    i = 0
    for filename in bias_files:
        bias_image += afwImage.ImageF(filename, amp_number_zero_based + 2)
        i += 1
        
    bias_image /= float(i)
    return bias_image
        
def GetImage_SingleAmp(filename, subtract_background = False, amp_number = 0):
    '''Return a single amplifier image, optionally background subtracted. Amplifiers are indexed 0-->15'''
    if subtract_background:
        config = MuonConfig()
        original_image = afwImage.DecoratedImageF(filename, amp_number + 2)
        maskedImg = afwImage.MaskedImageF(original_image.getImage())
        exposure = afwImage.ExposureF(maskedImg)
        bg, bgSubExp = estimateBackground(exposure, config.background, subtract=True)
        bgSubImg = bgSubExp.getMaskedImage().getImage()
        return bgSubImg
    else:
        return afwImage.ImageF(filename, amp_number + 2)
    return image
        


