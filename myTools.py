import numpy as np
import lsst.afw.display as afwDisplay
import lsst.afw.display.ds9 as ds9
import lsst.afw.image as afwImage


def argMax2d(array):
    """Get the index of the max value of an array.

    Actually for n dimensional, but easier to recall method if misnamed."""
    return np.unravel_index(np.argmax(array, axis=None), array.shape)


def countDetectedPixels(maskedImage):
    detectedBit = maskedImage.mask.getPlaneBitMask("DETECTED")
    return len(np.where(np.bitwise_and(maskedImage.mask.array, detectedBit))[0])


def boxcarAverage1DArray(data, boxcarLength):
    return np.convolve(data, np.ones((boxcarLength,))/boxcarLength, mode='valid')


def boxcarAverage2DArray(array, boxcarSize):
    if boxcarSize < 1:
        raise RuntimeError("Error - boxcar size cannot be less than 1")

    xsize, ysize = array.shape
    if boxcarSize == 1:
        return array

    ret = np.zeros((xsize - (boxcarSize - 1), ysize - (boxcarSize - 1)), dtype = np.float32)
    for x in range(xsize - boxcarSize + 1):
        for y in range(ysize - boxcarSize + 1):
            av = np.average(array[x:x+boxcarSize, y:y+boxcarSize])
            ret[x, y] = av
    return ret


def turnOffAllMasks(exceptFor=None):
    mpDict = afwImage.Mask().getMaskPlaneDict()
    for plane in mpDict.keys():
        if plane in exceptFor:
            continue
        ds9.setMaskPlaneColor(plane, afwDisplay.IGNORE)
