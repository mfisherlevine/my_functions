#!/usr/bin/env python

import argparse
import glob
import sys
from headerFunctions import keyValuesSetFromFiles


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

    keyValuesSetFromFiles(files, keys, joinKeys, noWarn)


if __name__ == '__main__':
    main()
