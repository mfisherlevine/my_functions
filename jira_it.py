#!/usr/bin/env python3
import requests
import argparse
import os
import subprocess
import re


def get_assignee(issue_data):
    assignee = None
    assignee_data = issue_data['fields'].get('assignee')
    assignee = assignee_data['displayName'] if assignee_data else 'Unassigned'
    return assignee


def get_reviewers(issue_data):
    reviewer_names = []
    reviewer_datas = issue_data['fields'].get('customfield_10048')
    if reviewer_datas is None:
        return 'No reviewer'

    for reviewer_data in reviewer_datas:
        reviewer_names.append(reviewer_data['displayName'])
    reviewers = ' ,'.join(reviewer_names)
    return reviewers


def get_jira_issue(issue_key, email, api_token):
    url = f"https://rubinobs.atlassian.net/rest/api/3/issue/{issue_key}"
    auth = requests.auth.HTTPBasicAuth(email, api_token)
    headers = {
        "Accept": "application/json",
    }
    response = requests.get(url, auth=auth, headers=headers)
    if response.status_code == 200:
        issue_data = response.json()
        summary = issue_data['fields']['summary']
        status = issue_data['fields']['status']['name']
        assignee = get_assignee(issue_data)
        reviewers = get_reviewers(issue_data)
        return summary, status, assignee, reviewers, url
    else:
        print(f"Failed to fetch issue {issue_key}: {response.status_code}")
        return None, None, None, None, url


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs='*')
    parser.add_argument("-v", "--verbose", action='store_true', default=False)
    args = parser.parse_args()

    if not args.urls:  # no ticket specified so find out the current dir's ticket
        path = os.getcwd()
        branchCommand = f"git --git-dir={path}/.git --work-tree={path} symbolic-ref --short HEAD"
        branchName = subprocess.check_output(branchCommand.split(), universal_newlines=True)
        args.urls = [branchName.strip()]

    for u in args.urls:
        if 'DM-' in u:
            u = re.sub('^.*/', '', u)
            ss, st, ass, rev, fullUrl = get_jira_issue(u, None, None)
            if rev is None:
                print(f"{u}  {ss} <{st}>@{ass}")
            else:
                print(f"{u}  {ss} <{st}>@{ass} RR:{rev}")
            if args.verbose:
                print(f"         {fullUrl}")
