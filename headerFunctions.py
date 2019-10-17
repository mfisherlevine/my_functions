import logging
import astropy
from astropy.io import fits
import filecmp
import sys

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("headerFunctions")
# logger.setLevel(logging.INFO)


def buildHashAndHeaderDicts(fileList, dataSize=100, dataHdu=1):
    """For a list of files, build dicts of hashed data and headers.

    Parameters
    ----------
    fileList : `list` of `str`
        The fully-specified paths of the files to scrape

    dataSize : int
        The side-length of the first n x n section of the data HDU to hash,
        i.e. dataSize = 100 hashes the data[0:100, 0:100]

    Returns
    -------
    dataDict : `dict`
        A dict, keyed by the data hash, with the values being strings of the
    filename

    headersDict : `dict`
        A dict, keyed by filename, with the values being the full primary
    header.

    """
    s = slice(0, dataSize)
    dataDict = {}
    headersDict = {}
    for filenum, filename in enumerate(fileList):
        if len(fileList) > 1000 and filenum%1000 == 0:
            logger.info(f"Processed {filenum} of {len(fileList)} files...")
        with fits.open(filename) as f:
            headersDict[filename] = f[0].header
            h = hash(f[dataHdu].data[s, s].tostring())
            if h in dataDict.keys():
                collision = dataDict[h]
                logger.warn(f"Duplicate file (or hash collision!) for files {filename} and {collision}!")
                if filecmp.cmp(filename, collision):
                    logger.warn("Filecmp shows files are identical")
                else:
                    logger.warn("Filecmp shows files differ - a genuine hash collision?!")
            else:
                dataDict[h] = filename
    return dataDict, headersDict


def keyValuesSetFromFiles(fileList, keys, joinKeys, noWarn=False, printResults=True):
    """For a list of FITS files, get the set of values for the given keys.

    Parameters
    ----------
    fileList : `list` of `str`
        The fully-specified paths of the files to scrape

    keys : `list` of `str`
        The header keys to scrape

    joinKeys : `list` of `str`
        List of keys to concatenate when scraping, e.g. for a header with
        FILTER1 = SDSS_u and FILTER2 == NB_640nm
        this would return SDSS_u+NB_640nm
        Useful when looking for the actual set, rather than taking the product
        of all the individual values, as some combinations may never happen.
    """

    if keys:  # necessary so that -j works on its own
        values = {k: set() for k in keys}
    else:
        keys = []
        values = None

    if joinKeys:
        joinedValues = set()

    print(f"Scraping headers from {len(fileList)} files...")

    for filenum, filename in enumerate(fileList):
        if len(fileList) > 1000 and filenum%1000 == 0:
            print(f"Processed {filenum} of {len(fileList)} files...")

        with fits.open(filename) as f:
            primaryHDU = f[0]
            header = primaryHDU.header
            for key in keys:
                if key in header:
                    values[key].add(header[key])
                else:
                    if not noWarn:
                        logger.warning(f"{key} not found in header of {filename}")

            if joinKeys:
                jVals = None
                try:
                    jVals = [header[k] for k in joinKeys]
                except Exception:
                    if not noWarn:
                        logger.warning(f"One or more of the requested joinKeys not found in {filename}")
                if jVals:
                    joinedValues.add("+".join([str(v) if not isinstance(v, astropy.io.fits.card.Undefined)
                                              else "<BLANK_VALUE>" for v in jVals]))
    if printResults:
        if values is not None:
            for key in values.keys():
                print(f"Values found for header key {key}:")
                print(f"{values[key]}\n")

        if joinKeys:
            print(f"Values found when joining {joinKeys}:")
            print(f"{joinedValues}\n")

    if joinKeys:
        return values, joinedValues

    return values
