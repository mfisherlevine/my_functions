#!/usr/bin/env python3

import re
import requests as rq
import json
import argparse
from bs4 import BeautifulSoup
import os
import subprocess

def check_url(url=None):
    if url is not None:
        waybackUrl = "https://jira.lsstcorp.org/browse/" + url
        R = rq.get(waybackUrl)
        BS = BeautifulSoup(R.text, 'lxml')
        #        print(R.content)

        tmp = BS.find_all("h1", {'id': 'summary-val'})
        summary = tmp[0].text.rstrip().lstrip()
        tmp = BS.find_all("span", {'id': 'status-val'})
        stat = tmp[0].text.rstrip().lstrip()
        tmp = BS.find_all("span", {'id': 'assignee-val'})
        assignee = tmp[0].text.rstrip().lstrip()
        tmp = BS.find_all("span", {'data-name': "Reviewers"})

        reviewer = None
        if len(tmp) > 0:
            tmp = tmp[0].find_all("span", {'class': 'user-hover'})
            tmp = [val.text.lstrip().rstrip() for val in tmp]
            reviewer = ", ".join(tmp)
        return summary, stat, assignee, reviewer, waybackUrl


parser = argparse.ArgumentParser()
parser.add_argument("urls", nargs='*')
parser.add_argument("-v", "--verbose", action='store_true', default=False)
args = parser.parse_args()

if not args.urls:  # no ticket specified so find out the current dir's ticket
    path = os.getcwd()
    branchCommand = f"git --git-dir={path}/.git --work-tree={path} symbolic-ref --short HEAD"
    branchName = subprocess.check_output(branchCommand.split(), universal_newlines=True)
    args.urls = branchName

for u in args.urls:
    if 'DM-' in u:
        u = re.sub('^.*/', '', u)
        ss, st, ass, rev, fullUrl = check_url(u)
        if rev is None:
            print(f"{u}  {ss} <{st}>@{ass}")
        else:
            print(f"{u}  {ss} <{st}>@{ass} RR:{rev}")
        if args.verbose:
            print(f"         {fullUrl}")
