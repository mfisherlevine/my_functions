import logging
import astropy
from astropy.io import fits
import filecmp
import sys
import os
import pickle

# redirect logger to stdout so that logger messages appear in notebooks too
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("headerFunctions")


def loadHeaderDictsFromLibrary(libraryFilename):
    """Load the header and hash dicts from a pickle file.

    Parameters
    ----------
    libraryFilename : `str`
        Path of the library file to load from

    Returns
    -------
    headersDict : `dict`
        A dict, keyed by filename, with the values being the full primary
    header, exactly as if it were built by buildHashAndHeaderDicts().

    dataDict : `dict`
        A dict, keyed by the data hash, with the values being strings of the
    filename, exactly as if it were built by buildHashAndHeaderDicts().
    """
    try:
        with open(libraryFilename, "rb") as pickleFile:
            headersDict, dataDict = pickle.load(pickleFile)

        if len(headersDict) != len(dataDict):
            print("Loaded differing numbers of entries for the header and data dicts.")
            print(f"{len(headersDict)} vs {len(dataDict)}")
            print("There were likely hash collisions in generating these library files.")
        else:
            print(f"Loaded {len(headersDict)} values from pickle files")
    except Exception as e:
        if not os.path.exists(libraryFilename):
            print(f"{libraryFilename} not found. If building the header dicts for the first time this"
                  " is to be expected.\nOtherwise you've misspecified the path to you library!")
        else:
            print(f"Something more sinister went wrong loading headers from {libraryFilename}:\n{e}")
        return {}, {}

    return headersDict, dataDict  # TODO: reorder these, or the builder function, so that they agree


def _saveToLibrary(libraryFilename, headersDict, dataDict):
    try:
        with open(libraryFilename, "wb") as dumpFile:
            pickle.dump((headersDict, dataDict), dumpFile, pickle.HIGHEST_PROTOCOL)
    except Exception:
        print("Failed to write pickle file! Here's a debugger so you don't lose all your work:")
        import ipdb as pdb
        pdb.set_trace()


def buildHashAndHeaderDicts(fileList, dataSize=100, dataHdu=1, libraryLocation=None):
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
    headersDict = {}
    dataDict = {}

    if libraryLocation:
        headersDict, dataDict = loadHeaderDictsFromLibrary(libraryLocation)

    # don't load files we already know about from the library
    fileList = [f for f in fileList if f not in headersDict.keys()]

    s = slice(0, dataSize)
    for filenum, filename in enumerate(fileList):
        if len(fileList) > 1000 and filenum%1000 == 0:
            logger.info(f"Processed {filenum} of {len(fileList)} files...")
        with fits.open(filename) as f:
            try:
                headersDict[filename] = f[0].header
                h = hash(f[dataHdu].data[s, s].tostring())
                if h in dataDict.keys():
                    collision = dataDict[h]
                    logger.warn(f"Duplicate file (or hash collision!) for files {filename} and {collision}!")
                    if filecmp.cmp(filename, collision):
                        logger.warn("Filecmp shows files are identical")
                    else:
                        logger.warn("Filecmp shows files differ - "
                                    "likely just zeros for data (or a genuine hash collision!)")
                else:
                    dataDict[h] = filename
            except Exception:
                logger.warn(f"Failed to load {filename} - file is likely corrupted.")

    # we have always added to this, so save it back over the original
    if libraryLocation and len(fileList) > 0:
        _saveToLibrary(libraryLocation, headersDict, dataDict)

    return dataDict, headersDict


def sorted(inlist, replacementValue="<BLANK VALUE>"):
    """Redefinition of sorted() to deal with blank values and str/int mixes"""
    from builtins import sorted as _sorted
    output = [str(x) if not isinstance(x, astropy.io.fits.card.Undefined)
              else replacementValue for x in inlist]
    output = _sorted(output)
    return output


def keyValuesSetFromFiles(fileList, keys, joinKeys, noWarn=False, printResults=True, libraryLocation=None):
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
    print(f"Scraping headers from {len(fileList)} files...")

    hashDict, headerDict = buildHashAndHeaderDicts(fileList, libraryLocation=libraryLocation)

    if keys:  # necessary so that -j works on its own
        kValues = {k: set() for k in keys}
    else:
        keys = []
        kValues = None

    if joinKeys:
        joinedValues = set()

    for filename in headerDict.keys():
        header = headerDict[filename]
        for key in keys:
            if key in header:
                kValues[key].add(header[key])
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
                # substitute <BLANK_VALUE> when there is an undefined card
                # because str(v) will give the address for each blank value
                # too, meaning each blank card looks like a different value
                joinedValues.add("+".join([str(v) if not isinstance(v, astropy.io.fits.card.Undefined)
                                          else "<BLANK_VALUE>" for v in jVals]))

    if printResults:
        if kValues is not None:
            for key in kValues.keys():
                print(f"Values found for header key {key}:")
                print(f"{sorted(kValues[key])}\n")

        if joinKeys:
            print(f"Values found when joining {joinKeys}:")
            print(f"{sorted(joinedValues)}\n")

    if joinKeys:
        return kValues, joinedValues

    return kValues


def compareHeaders(filename1, filename2):
    """Compare the headers of two files in detail.

    First, the two files are confirmed to have the same pixel data to ensure
    the files should be being compared (by hashing the first 100x100 pixels
    in HDU 1).

    It then prints out:
        the keys that appear in A and not B
        the keys that appear in B but not A
        the keys that in common, and of those in common:
            which are the same,
            which differ,
            and where different, what the differing values are

    Parameters
    ----------
    filename1 : str
        Full path to the first of the files to compare

    filename2 : str
        Full path to the second of the files to compare
    """
    assert isinstance(filename1, str)
    assert isinstance(filename2, str)

    hashDict1, headerDict1 = buildHashAndHeaderDicts([filename1])
    hashDict2, headerDict2 = buildHashAndHeaderDicts([filename2])

    if list(hashDict1.keys())[0] != list(hashDict2.keys())[0]:
        print("Pixel data was not the same - did you really mean to compare these files?")
        print(f"{filename1}\n{filename2}")
        cont = input("Press y to continue, anything else to quit:")
        if cont.lower()[0] != 'y':
            exit()

    # you might think you don't want to always call sorted() on the key sets
    # BUT otherwise they seem to be returned in random order each time you run
    # and that can be crazy-making

    h1 = headerDict1[filename1]
    h2 = headerDict2[filename2]
    h1Keys = list(h1.keys())
    h2Keys = list(h2.keys())

    commonKeys = set(h1Keys)
    commonKeys = commonKeys.intersection(h2Keys)

    keysInh1NotInh2 = sorted([_ for _ in h1Keys if _ not in h2Keys])
    keysInh2NotInh1 = sorted([_ for _ in h2Keys if _ not in h1Keys])

    print(f"Keys in {filename1} not in {filename2}:\n{keysInh1NotInh2}\n")
    print(f"Keys in {filename2} not in {filename1}:\n{keysInh2NotInh1}\n")
    print(f"Keys in common:\n{sorted(commonKeys)}\n")

    # put in lists so we can output neatly rather than interleaving
    identical = []
    differing = []
    for key in commonKeys:
        if h1[key] == h2[key]:
            identical.append(key)
        else:
            differing.append(key)

    assert len(identical)+len(differing) == len(commonKeys)

    if len(identical) == len(commonKeys):
        print("All keys in common have identical values :)")
    else:
        print("Of the common keys, the following had identical values:")
        print(f"{sorted(identical)}\n")
        print("Common keys with differing values were:")
        for key in sorted(differing):
            d = "<blank card>".ljust(25)
            v1 = str(h1[key]).ljust(25) if not isinstance(h1[key], astropy.io.fits.card.Undefined) else d
            v2 = str(h2[key]).ljust(25) if not isinstance(h2[key], astropy.io.fits.card.Undefined) else d
            print(f"{key.ljust(8)}: {v1} vs {v2}")
