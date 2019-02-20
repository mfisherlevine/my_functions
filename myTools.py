import numpy as np


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
