#!/usr/bin/env python
import numpy, argparse, re
from lsst.afw.cameraGeom.fitsUtils import DetectorBuilder, setByKey, getByKey
from lsst.ip.isr import AssembleCcdTask
import lsst.afw.math as afwMath
import lsst.afw.cameraGeom.utils as camGeomUtils
import lsst.afw.image as afwImage
from lsst.pex.exceptions import NotFoundError

class TestCamDetectorBuilder(DetectorBuilder):
    def _sanitizeHeaderMetadata(self, metadata, clobber):
        naxis1 = getByKey(metadata, 'NAXIS1')
        naxis2 = getByKey(metadata, 'NAXIS2')
        biassec = getByKey(metadata, 'BIASSEC')
        if biassec is None:
        #If BIASSEC doesn't exist, set a default
            setByKey(metadata, 'BIASSEC', '[523:544, 1:2002]', clobber)
        #Get channel number and convert to zero index
        channel = getByKey(metadata, 'CHANNEL')-1
        if channel is None:
            raise ValueError("Channel keyword not found in header")
        ny = channel//8
        sign = 1 if ny%2 else -1
        nx = 7*ny - sign*(channel%8)
        setByKey(metadata, 'DTV1', nx*naxis1, clobber)
        setByKey(metadata, 'DTV2', ny*naxis2, clobber)
        try:
            #map to the keyword expected for this value
            setByKey(metadata, 'DTM1_1', metadata.get('LTM1_1'), clobber)
            setByKey(metadata, 'DTM2_2', metadata.get('LTM2_2'), clobber)
        except NotFoundError:
            #Try to set the DTM1_1 and DTM2_2 values from the DETSEC
            detsec = getByKey(metadata, 'DETSEC')
            _, x0, x1, y0, y1, _ = re.split('\[|:|,|\]', detsec)
            dtm1_1 = -1 if int(x0) > int(x1) else 1
            dtm2_2 = -1 if int(y0) > int(y1) else 1
            setByKey(metadata, 'DTM1_1', dtm1_1, clobber)
            setByKey(metadata, 'DTM2_2', dtm2_2, clobber)
        self._defaultSanitization(metadata, clobber)

class imageSource(object):
    def __init__(self, exposure):
        self.exposure = exposure
        self.image = self.exposure.getMaskedImage().getImage()
    def getCcdImage(self, det, imageFactory, binSize):
        return afwMath.binImage(self.image, binSize)

def AssembleImage(input_file, doGainCorrection = False, trim = True):
    import os
    
    imDict = {}
    afilelist = []
    dfilename = '%s[%s]' % (input_file, 0)
    
    for i in range(16):
        filename = '%s[%i]' % (input_file, i+1)
        md = afwImage.readMetadata(filename)
        afilelist.append(filename)
        imDict[md.get('EXTNAME')] = afwImage.ImageF(filename)
        
    db = TestCamDetectorBuilder(dfilename, afilelist, inAmpCoords=True, clobberMetadata=True)
    det = db.buildDetector()
    assembleInput = {}
    for amp in det:
        im = imDict[amp.getName()]
        oscanim = im.Factory(im, amp.getRawHorizontalOverscanBBox())
        oscan = numpy.median(oscanim.getArray())
        imArr = im.getArray()
        imArr -= oscan
        #Calculate and correct for gain
        if doGainCorrection:
            # Buffer so edge rolloff doesn't interfere
            buffImArr = imArr[30:-30][30:-30]
            medCounts = numpy.median(buffImArr)
            stdCounts = numpy.std(buffImArr)
            gain = medCounts/stdCounts**2
            imArr *= gain
        assembleInput[amp.getName()] = db.makeExposure(im)
    assembleConfig = AssembleCcdTask.ConfigClass()

    if trim:
        assembleConfig.doTrim = True
        assembler = AssembleCcdTask(config=assembleConfig)
        resultExp = assembler.assembleCcd(assembleInput)
#         camGeomUtils.showCcd(resultExp.getDetector(), imageSource(resultExp), frame=0)
        return resultExp

    else:
        assembleConfig.doTrim = False
        assembler = AssembleCcdTask(config=assembleConfig)
        resultExp = assembler.assembleCcd(assembleInput)
#         camGeomUtils.showCcd(resultExp.getDetector(), imageSource(resultExp), frame=1)
        return resultExp
