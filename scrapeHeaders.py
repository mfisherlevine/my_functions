#!/usr/bin/env python

import logging
import argparse
import glob
import sys
from astropy.io import fits


def scrapeKeys(fileList, keys, noWarn=False):
    values = {k: set() for k in keys}

    print(f"Scraping headers from {len(fileList)} files...")

    for filename in fileList:
        with fits.open(filename) as f:
            primaryHDU = f[0]
            header = primaryHDU.header
            for key in keys:
                if key in header:
                    values[key].add(header[key])
                else:
                    if not noWarn:
                        logging.warning(f"{key} not found in header of {filename}")

    for key in values.keys():
        print(f"Values found for header key {key}:")
        print(f"{values[key]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, help="List of files to scrape")
    parser.add_argument("-k", metavar='keys', dest='keys', nargs='+', type=str,
                        help="Keys to return")
    parser.add_argument("--noWarn", action='store_true', help="Supress warnings for keys not in header?",
                        default=False, dest='noWarn')

    args = parser.parse_args()
    files = glob.glob(args.files)
    keys = args.keys
    noWarn = args.noWarn

    if not files:
        print('Found no files matching: ' + args.files, file=sys.stderr)
        sys.exit(1)

    scrapeKeys(files, keys, noWarn)


if __name__ == '__main__':
    main()
