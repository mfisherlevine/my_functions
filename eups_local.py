import subprocess
import numpy as np
import sys
import re
from jira_it import check_url

GITMAP = {
    "nothing to commit, working tree clean": "✅",
    "Untracked files:": "🤷‍♂️ untracked files",
    "Changes not staged for commit:": "❌ unstaged work",
    "Parsing failed": "⛔️",
    "diverged": "🔀 diverged",
    # detached done differently due to detach point in value
}


def show(text):
    for line in text.split('\n'):
        print(line)


def print_results(*, packages, paths, branches, statuses, ticketDetails, sorting=''):
    """Print the results

    Default sorting is alphabetical by package name (also 'p')
    Use 't' to sort by ticket
    Use 's' to sort by status
    """
    packages = np.array(packages)
    paths = np.array(paths)
    branches = np.array(branches)
    statuses = np.array(statuses)
    ticketDetails = np.array(ticketDetails)

    if sorting == '' or sorting == 'p':
        inds = packages.argsort()
    elif sorting == 't':
        inds = branches.argsort()
    elif sorting == 's':
        inds = statuses.argsort()[::-1]  # bad stuff first
    else:
        inds = range(len(packages))

    paddings = []
    paddings.append(max([len(x) for x in packages]) + 2)
    paddings.append(max([len(x) for x in paths]) + 2)
    paddings.append(max([len(x) for x in branches]) + 3)
    for package, path, branch, status, details in zip(packages[inds],
                                                      paths[inds],
                                                      branches[inds],
                                                      statuses[inds],
                                                      ticketDetails[inds]):
        print(f"{package:{paddings[0]}} {path:{paddings[1]}} {branch:{paddings[2]}} {status}")
        if details != '':
            print(f'\t {details}')


def fetchAndCheckMain(path):
    """
    Examples:

    On branch main
    Your branch is up-to-date with 'origin/main'.
    nothing to commit, working tree clean


    On branch main
    Your branch is behind 'origin/main' by 4 commits, and can be fast-forwarded
      (use "git pull" to update your local branch)
    nothing to commit, working tree clean
    """
    fetchCmd = f"git --git-dir={path}/.git --work-tree={path} fetch"
    _ = subprocess.check_output(fetchCmd.split(), universal_newlines=True)
    statusCmd = f"git --git-dir={path}/.git --work-tree={path} status"
    newGitOutput = subprocess.check_output(statusCmd.split(), universal_newlines=True)

    line2 = newGitOutput.split('\n')[1]

    if line2 == "Your branch is up to date with 'origin/main'.":
        line3 = newGitOutput.split('\n')[3]
        if line3 in GITMAP.keys():
            return GITMAP[line3]
        else:
            return GITMAP["Parsing failed"]
    if line2.startswith("Your branch is behind 'origin/main' by"):
        line2 = line2.replace("Your branch is behind 'origin/main' by ", "")
        n = line2.split()[0]
        status = f"✅ ⬇️  {n} commits"
        return status

    return "??? - main parse fail"


def parseGitOutput(gitOutput, path):
    #  Should maybe switch to using --porcelain for the initial status stuff

    branch = gitOutput.split('\n')[0].split()[2]  # always the third word of first line?

    if branch == "at":
        line1 = gitOutput.split('\n')[0]
        line2 = gitOutput.split('\n')[1]

        if line1.startswith('HEAD detached'):  # check it is really detached
            branch = "n/a - detached"
            status = f'🔪 at {line1.split()[3]}'
            if line2 == "nothing to commit, working tree clean":
                status += " ✅"
            else:
                status += " ❌"
            return branch, status

    if branch == 'main':
        status = fetchAndCheckMain(path)
        return branch, status

    try:
        line3 = gitOutput.split('\n')[3]
        if line3.endswith('respectively.'):
            return branch, GITMAP['diverged']

        status = GITMAP[line3]
    except (KeyError, IndexError):
        print(f"Failed to map the following git status for branch {branch}:")
        print('---------')
        show(gitOutput)
        print('---------')
        status = "⛔️"
    return branch, status


def getLocalPackagesFromEupsOutput(eupsOutput):
    lines = [line for line in eupsOutput.split('\n') if "LOCAL" in line]
    packages, paths = [], []

    for line in lines:
        ret = line.split()
        assert len(ret) == 3
        if ret[0] == 'eups':  # rubin env now has eups as LOCAL for cloned scipipe
            continue
        packages.append(ret[0])
        paths.append(ret[1][6:])

    return packages, paths


def getBranchAndStatus(package, path):
    cmd = f"git --git-dir={path}/.git --work-tree={path} status"
    gitOutput = subprocess.check_output(cmd.split(), universal_newlines=True)
    branch, status = parseGitOutput(gitOutput, path)

    return branch, status


def dumpGITMAP():
    for k, v in GITMAP.items():
        print(f"{v} : {k}")


if __name__ == "__main__":
    # worst argparser ever, but - and -- are both OK, and command just has to
    # begin with the right letter. It was quicker than remembering how to use
    # argparse properly - don't judge me.

    args = sys.argv
    sorting = ''
    printTicketDetails = False
    if len(args) > 1:
        sorting = args[1]
        sorting = sorting.replace('-', '')
        if 'v' in sorting:
            printTicketDetails = True
        sorting = sorting[0]
        if sorting == '?':
            dumpGITMAP()
            exit()
        assert sorting in ['p', 's', 't']

    cmd = 'eups list -s'
    eupsOutput = subprocess.check_output(cmd.split(), universal_newlines=True)

    packages, paths = getLocalPackagesFromEupsOutput(eupsOutput)
    if len(packages) == 0:
        print("No local packages were found to be setup. If you think this is a lie try")
        print("eups list -s | grep LOCAL")
        exit()

    branches, statuses, ticketDetails = [], [], []
    for package, path in zip(packages, paths):
        if '/sdf/group/rubin/g/shared' in path:
            branches.append('main')
            statuses.append('--shared stack--')
            ticketDetails.append(None)
            continue
        branch, status = getBranchAndStatus(package, path)
        branches.append(branch)
        statuses.append(status)

        if printTicketDetails:
            try:
                if 'DM-' in branch:
                    branch = re.sub('^.*/', '', branch)
                    result = check_url(branch)
                    ticketDetails.append(result[0])  # [0] is the description
                else:
                    ticketDetails.append('')
            except Exception:
                ticketDetails.append('Failed to contact Jira')
        else:
            ticketDetails.append('')

    assert len(packages) == len(paths) == len(branches) == len(statuses) == len(ticketDetails)
    print_results(packages=packages,
                  paths=paths,
                  branches=branches,
                  statuses=statuses,
                  ticketDetails=ticketDetails,
                  sorting=sorting)
