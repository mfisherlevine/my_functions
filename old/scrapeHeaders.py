#!/usr/bin/env python

import logging
import argparse
import glob
import sys
from astropy.io import fits


def scrapeKeys(fileList, keys, joinKeys, noWarn=False):
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
                        logging.warning(f"{key} not found in header of {filename}")

            if joinKeys:
                jVals = None
                try:
                    jVals = [header[k] for k in joinKeys]
                except Exception:
                    if not noWarn:
                        logging.warning(f"One or more of the requested joinKeys not found in {filename}")
                if jVals:
                    joinedValues.add("+".join(jVals))

    if values is not None:
        for key in values.keys():
            print(f"Values found for header key {key}:")
            print(f"{values[key]}\n")

    if joinKeys:
        print(f"Values found when joining {joinKeys}:")
        print(f"{joinedValues}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, help=("List of files to scrape. Enclose any glob "
                                                 "patterns in quotes so they are passed unexpanded"))
    parser.add_argument("-k", metavar='keys', dest='keys', nargs='+', type=str,
                        help="Keys to return")
    parser.add_argument("-j", metavar='joinKeys', dest='joinKeys', nargs='+', type=str,
                        help="Keys to return joined together.")
    parser.add_argument("--noWarn", action='store_true', help="Suppress warnings for keys not in header?",
                        default=False, dest='noWarn')

    args = parser.parse_args()
    files = glob.glob(args.files)
    keys = args.keys
    joinKeys = args.joinKeys
    noWarn = args.noWarn

    if not keys and not joinKeys:
        print(("No keys requested for scraping! Specify with e.g. -k KEY1 KEY2, "
               "or with e.g. -j FILTER FILTER2 for keys to join"), file=sys.stderr)
        sys.exit(1)

    if not files:
        print('Found no files matching: ' + args.files, file=sys.stderr)
        sys.exit(1)

    scrapeKeys(files, keys, joinKeys, noWarn)


if __name__ == '__main__':
    main()
