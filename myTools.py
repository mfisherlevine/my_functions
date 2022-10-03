import os
import math
import threading
import subprocess
import tempfile
import shutil
import itertools

import numpy as np
import matplotlib.pyplot as plt

import lsst.afw.math as afwMath
import lsst.afw.display as afwDisplay
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.daf.persistence.butlerExceptions as butlerExcept
import lsst.meas.algorithms as measAlg
import lsst.rapid.analysis.butlerUtils as bu

# import lsst.log
# from contextlib import redirect_stdout, redirect_stderr

CALIB_VALUES = ['FlatField position', 'Park position', 'azel_target']


def getLatissOnSkyDataIds_gen2(butler, skipTypes=['BIAS', 'DARK', 'FLAT'], checkObject=True, startDate=None):
    """Return a dayObs, seqNum dict for each on-sky observation"""
    def isOnSky(dataId):
        imageType = dataId.pop('imageType')  # want it gone for later
        obj = dataId.pop('object')  # want it gone for later
        if obj == 'NOTSET' and checkObject:
            return False
        if imageType not in skipTypes:
            return True
        return False

    days = butler.queryMetadata('raw', 'dayObs')
    days = [d for d in days if d.startswith('202')]  # went on sky in Jan 2020
    if startDate:
        days = [d for d in days if d >= startDate]
    
    allDataIds = []
    for day in days:
        ret = butler.queryMetadata('raw', ['seqNum', 'imageType', 'object'], dayObs=day)
        allDataIds.extend([{'dayObs': day, 'seqNum': s, 'imageType': i, 'object': o} for (s, i, o) in ret])

    return [d for d in filter(isOnSky, allDataIds)]


def getLatissOnSkyDataIds(butler, skipTypes=['bias', 'dark', 'flat'], checkObject=True, expanded=False,
                          startDate=None, endDate=None):
    def isOnSky(expRecord):
        imageType = expRecord.observation_type
        obj = expRecord.target_name
        if checkObject and obj == 'NOTSET':
            return False
        if imageType not in skipTypes:
            return True
        return False
    
    recordSets = []
    days = bu.getDaysOnSky(butler)
    if startDate:
        days = [d for d in days if d >= startDate]
    if endDate:
        days = [d for d in days if d <= startDate]
        
    days = sorted(set(days))

    where = "exposure.day_obs=day_obs"
    for day in days:
        records = butler.registry.queryDimensionRecords("exposure", where=where, bind={'day_obs': day})
        recordSets.append(records)

    dataIds = [r.dataId for r in filter(isOnSky, itertools.chain(*recordSets))]
    if expanded:
        return [butler.registry.expandDataId(dataId) for dataId in dataIds]
    else:
        return dataIds


def argMax2d(array):
    """Get the index of the max value of an array.

    Actually for n dimensional, but easier to recall method if misnamed."""
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def countPixels(maskedImage, maskPlane):
    bit = maskedImage.mask.getPlaneBitMask(maskPlane)
    return len(np.where(np.bitwise_and(maskedImage.mask.array, bit))[0])


def countDetectedPixels(maskedImage):
    return countPixels(maskedImage, "DETECTED")


def boxcarAverage1DArray(data, boxcarLength):
    return np.convolve(data, np.ones((boxcarLength,))/boxcarLength, mode='valid')


def boxcarAverage2DArray(array, boxcarSize):
    if boxcarSize < 1:
        raise RuntimeError("Error - boxcar size cannot be less than 1")

    xsize, ysize = array.shape
    if boxcarSize == 1:
        return array

    ret = np.zeros((xsize - (boxcarSize - 1), ysize - (boxcarSize - 1)), dtype=np.float32)
    for x in range(xsize - boxcarSize + 1):
        for y in range(ysize - boxcarSize + 1):
            av = np.average(array[x:x+boxcarSize, y:y+boxcarSize])
            ret[x, y] = av
    return ret


def isExposureTrimmed(exp):
    det = exp.getDetector()
    if exp.getDimensions() == det.getBBox().getDimensions():
        return True
    return False


def displayArray(arrayData, frame=0):
    tempIm = afwImage.ImageF(arrayData)
    afwDisplay.Display(frame=frame).mtv(tempIm)


def disp_turnOffAllMasks(exceptFor=None):
    mpDict = afwImage.Mask().getMaskPlaneDict()
    for plane in mpDict.keys():
        if plane in exceptFor:
            continue
        ds9.setMaskPlaneColor(plane, afwDisplay.IGNORE)


def invertDictionary(inputDict):
    return dict((v, k) for (k, v) in inputDict.items())

# def disp_turnOffAllMasks(exceptFor=None):
#     maskPlanes = afwImage.Mask().getMaskPlaneDict().keys()
#     ignorePlanes = [p for p in maskPlanes if p not in exceptFor]
#     for plane in ignorePlanes:
#         ds9.setMaskPlaneColor(plane, afwDisplay.IGNORE)


# def disp_setMyMaskColors():
# dispI.setMaskPlaneColor("CROSSTALK", afwDisplay.ORANGE)
# dispI.setMaskPlaneColor("CROSSTALK", "ignore")

def binCentersFromBinEdges(binEdges):
    """Get the bin centers from the histogram bin edges

    Parameters
    ----------

    binEdges : `np.array` of `float`
        List of the binEdges as generated by np.hist().

    Returns
    -------

    binCenters : `np.array` of `float`
        List of the corresponding bin centers.
    """
    binCenters = []
    for i in range(len(binEdges)-1):
        binCenters.append((binEdges[i] + binEdges[i+1])/2)
    return np.array(binCenters)


# Useful one-liners ###############

def setMaskTransparency():
    afwDisplay.setDefaultMaskTransparency(85)


def expToPng(exp, saveFilename, title=None):
    fig = plt.figure(figsize=(15, 15))
    afwDisplay.setDefaultBackend("matplotlib")
    display = afwDisplay.Display(fig, open=True)
    display.setImageColormap('viridis')
    display.scale('asinh', 'zscale')
    display.mtv(exp, title=title)
    plt.tight_layout()
    fig.savefig(saveFilename)
    return


def smoothExp(exp, smoothing, kernelSize=7):
    """Use for DISPLAY ONLY!
    Return a smoothed copy of the exposure with the original mask plane in place."""
    psf = measAlg.DoubleGaussianPsf(kernelSize, kernelSize, smoothing/(2*math.sqrt(2*math.log(2))))
    newExp = exp.clone()
    originalMask = exp.mask

    kernel = psf.getKernel()
    afwMath.convolve(newExp.maskedImage, newExp.maskedImage, kernel, afwMath.ConvolutionControl())
    newExp.mask = originalMask
    return newExp


class LogRedirect:
    def __init__(self, fd, dest, encoding="utf-8", errors="strict"):

        # Save original handle so we can restore it later.
        self.saved_handle = os.dup(fd)
        self.saved_fd = fd
        self.saved_dest = dest

        # Redirect `fd` to the write end of the pipe.
        pipe_read, pipe_write = os.pipe()
        os.dup2(pipe_write, fd)
        os.close(pipe_write)

        # This thread reads from the read end of the pipe.
        def consumer_thread(f, data):
            while True:
                buf = os.read(f, 1024)
                if not buf:
                    break
                data.write(buf.decode(encoding, errors))
            os.close(f)
            return

        # Spawn consumer thread, and give it a mutable `data` item to
        # store the redirected output.
        self.thread = threading.Thread(target=consumer_thread, args=(pipe_read, dest))
        self.thread.start()

    def finish(self):
        # Cleanup: flush streams, restore `fd`
        self.saved_dest.flush()
        os.dup2(self.saved_handle, self.saved_fd)
        os.close(self.saved_handle)
        self.thread.join()

# import sys
# lr = LogRedirect(1, sys.stdout)


def animateDataIds(dataIds, pathToPngs, outFilename, clobber=True, copyGifToOutdir=True,
                   ffMpegBinary='/home/mfl/bin/ffmpeg'):
    def dataIdsToFileList(dataIds, pathToPngs):
        filenames = []
        for d in dataIds:
            f = os.path.join(pathToPngs, f"dayObs_{d['dayObs']}seqNum_{str(d['seqNum']).zfill(3)}.png")
            if os.path.exists(f):
                filenames.append(f)
            else:
                print(f"Failed to find {f} for {d}")
        return filenames

    fileList = dataIdsToFileList(dataIds, pathToPngs)
    tempFilename = tempfile.mktemp('.gif')
    subprocess.run(['convert', '-delay', '10', '-loop', '0', *fileList, tempFilename], check=True)

    if os.path.exists(outFilename):
        if clobber:
            os.remove(outFilename)
        else:
            raise RuntimeError(f'Output file {outFilename} exists and clobber==False!')

    if copyGifToOutdir:
        gifName = os.path.splitext(outFilename)[0] + '.gif'
        shutil.copy(tempFilename, gifName)
    assert os.path.exists(ffMpegBinary)
    command = (f'{ffMpegBinary} -i {tempFilename} -pix_fmt yuv420p'
               f' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {outFilename}')
    output, error = subprocess.Popen(command, universal_newlines=True, shell=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()


def summarizeVisit(butler, *, exp=None, extendedSummary=False, **kwargs):
    from astroquery.simbad import Simbad
    import astropy.units as u
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz

    def _airMassFromrRawMd(md):
        auxTelLocation = EarthLocation(lat=-30.244639*u.deg, lon=-70.749417*u.deg, height=2663*u.m)
        time = Time(md['DATE-OBS'])
        skyLocation = SkyCoord(md['RASTART'], md['DECSTART'], unit=u.deg)
        altAz = AltAz(obstime=time, location=auxTelLocation)
        observationAltAz = skyLocation.transform_to(altAz)
        return observationAltAz.secz.value

    items = ["OBJECT", "expTime", "FILTER", "imageType"]
    obj, expTime, filterCompound, imageType = butler.queryMetadata('raw', items, **kwargs)[0]
    filt, grating = filterCompound.split('~')
    rawMd = butler.get('raw_md', **kwargs)
    airmass = _airMassFromrRawMd(rawMd)

    print(f"Object name: {obj}")
    print(f"expTime:     {expTime}s")
    print(f"imageType:   {imageType}")
    print(f"Filter:      {filt}")
    print(f"Grating:     {grating}")
    print(f"Airmass:     {airmass:.3f}")

    if imageType not in ['BIAS', 'FLAT', 'DARK']:
        simbadObj = Simbad.query_object(obj)
        if simbadObj is None:
            print(f"Failed to find {obj} in Simbad.")
        else:
            assert(len(simbadObj.as_array()) == 1)
            raStr = simbadObj[0]['RA']
            decStr = simbadObj[0]['DEC']
            skyLocation = SkyCoord(raStr, decStr, unit=(u.hourangle, u.degree), frame='icrs')
            raRad, decRad = skyLocation.ra.rad, skyLocation.dec.rad
            print(f"obj RA  (str):   {raStr}")
            print(f"obj DEC (str):   {decStr}")
            print(f"obj RA  (rad):   {raRad:5f}")
            print(f"obj DEC (rad):   {decRad:5f}")
            print(f"obj RA  (deg):   {raRad*180/math.pi:5f}")
            print(f"obj DEC (deg):   {decRad*180/math.pi:5f}")

            if exp is not None:  # calc source coords from exp wcs
                ra = geom.Angle(raRad)
                dec = geom.Angle(decRad)
                targetLocation = geom.SpherePoint(ra, dec)
                pixCoord = exp.getWcs().skyToPixel(targetLocation)
                print(exp.getWcs())
                print(f"Source location: {pixCoord} using exp provided")
            else:  # try to find one, but not too hard
                datasetTypes = ['calexp', 'quickLookExp', 'postISRCCD']
                for datasetType in datasetTypes:
                    wcs = None
                    typeUsed = None
                    try:
                        wcs = butler.get(datasetType + '_wcs', **kwargs)
                        typeUsed = datasetType
                        break
                    except butlerExcept.NoResults:
                        pass
                if wcs is not None:
                    ra = geom.Angle(raRad)
                    dec = geom.Angle(decRad)
                    targetLocation = geom.SpherePoint(ra, dec)
                    pixCoord = wcs.skyToPixel(targetLocation)
                    print(wcs)
                    print(f"Source location: {pixCoord} using {typeUsed}")

    if extendedSummary:
        print('\n--- Extended Summary ---')
        ranIsr = False
        if exp is None:
            print("Running isr to compute image stats...")

            # catch all the ISR chat
            # logRedirection1 = LogRedirect(1, open(os.devnull, 'w'))
            # logRedirection2 = LogRedirect(2, open(os.devnull, 'w'))
            # import ipdb as pdb; pdb.set_trace()
            from lsst.ip.isr.isrTask import IsrTask
            isrConfig = IsrTask.ConfigClass()
            isrConfig.doLinearize = False
            isrConfig.doBias = False
            isrConfig.doFlat = False
            isrConfig.doDark = False
            isrConfig.doFringe = False
            isrConfig.doDefect = False
            isrConfig.doWrite = False
            isrTask = IsrTask(config=isrConfig)
            dataRef = butler.dataRef('raw', **kwargs)
            exp = isrTask.runDataRef(dataRef).exposure
            wcs = exp.getWcs()
            ranIsr = True
            # logRedirection1.finish()  # end re-direct
            # logRedirection2.finish()  # end re-direct

            print(wcs)
        if simbadObj and ranIsr:
            ra = geom.Angle(raRad)
            dec = geom.Angle(decRad)
            targetLocation = geom.SpherePoint(ra, dec)
            pixCoord = wcs.skyToPixel(targetLocation)
            print(f"Source location: {pixCoord} using postISR just-reconstructed wcs")

        print(f'\nImage stats from {"just-constructed" if ranIsr else "provided"} exp:\n')
        print(f'Image mean:   {np.mean(exp.image.array):.2f}')
        print(f'Image median: {np.median(exp.image.array):.2f}')
        print(f'Image min:    {np.min(exp.image.array):.2f}')
        print(f'Image max:    {np.max(exp.image.array):.2f}')
        # TODO: quartiles/percentiles here
        # number of masked pixels, saturated pixels

        print()
        print(f'BAD pixels:      {countPixels(exp.maskedImage, "BAD")}')
        print(f'SAT pixels:      {countPixels(exp.maskedImage, "SAT")}')
        print(f'CR pixels:       {countPixels(exp.maskedImage, "CR")}')
        print(f'INTRP pixels:    {countPixels(exp.maskedImage, "INTRP")}')
        print(f'DETECTED pixels: {countPixels(exp.maskedImage, "DETECTED")}')

        # detector = exp.getDetector()
        visitInfo = exp.getInfo().getVisitInfo()
        rotAngle = visitInfo.getBoresightRotAngle()
        boresight = visitInfo.getBoresightRaDec()
        md = butler.get('raw_md', **kwargs)

        print("\n From VisitInfo:")
        print(f"boresight: {boresight}")
        print(f"rotAngle:  {rotAngle}")
        print(f"  →        {rotAngle.asDegrees():.4f} deg")

        print("\n From raw_md:")
        print(f"ROTPA:     {md['ROTPA']} deg")
        print(f"  →        {(md['ROTPA']*math.pi/180):.6f} rad")


def mPrint(obj):
    try:
        import black
    except ImportError:
        print('Failed to import black, printing normally, sorry...')
        print(obj)
    print(black.format_str(repr(obj), mode=black.Mode()))
