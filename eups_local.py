import subprocess
import numpy as np
import sys

GITMAP = {
    "nothing to commit, working tree clean": "✅",
    "Untracked files:": "🤷‍♂️ untracked files",
    "Changes not staged for commit:": "❌ unstaged work",
    "Parsing failed": "⛔️",
    "diverged": "🔀 diverged",
}


def show(text):
    for line in text.split('\n'):
        print(line)


def print_results(packages, paths, branches, statuses, sorting=''):
    """Print the results

    Default sorting is alphabetical by package name (also 'p')
    Use 't' to sort by ticket
    Use 's' to sort by status
    """
    packages = np.array(packages)
    paths = np.array(paths)
    branches = np.array(branches)
    statuses = np.array(statuses)

    if sorting == '' or sorting == 'p':
        inds = packages.argsort()
    elif sorting == 't':
        inds = branches.argsort()
    elif sorting == 's':
        inds = statuses.argsort()[::-1]  # bad stuff first

    paddings = []
    paddings.append(max([len(x) for x in packages]) + 2)
    paddings.append(max([len(x) for x in paths]) + 2)
    paddings.append(max([len(x) for x in branches]) + 3)

    for package, path, branch, status in zip(packages[inds], paths[inds], branches[inds], statuses[inds]):
        print(f"{package:{paddings[0]}} {path:{paddings[1]}} {branch:{paddings[2]}} {status}")


def fetchAndCheckMaster(path):
    """
    Examples:

    On branch master
    Your branch is up-to-date with 'origin/master'.
    nothing to commit, working tree clean


    On branch master
    Your branch is behind 'origin/master' by 4 commits, and can be fast-forwarded.
      (use "git pull" to update your local branch)
    nothing to commit, working tree clean
    """
    fetchCmd = f"git --git-dir={path}/.git --work-tree={path} fetch"
    _ = subprocess.check_output(fetchCmd.split(), universal_newlines=True)
    statusCmd = f"git --git-dir={path}/.git --work-tree={path} status"
    newGitOutput = subprocess.check_output(statusCmd.split(), universal_newlines=True)

    line2 = newGitOutput.split('\n')[1]

    if line2 == "Your branch is up-to-date with 'origin/master'.":
        return "✅"
    if line2.startswith("Your branch is behind 'origin/master' by"):
        line2 = line2.replace("Your branch is behind 'origin/master' by ", "")
        n = line2.split()[0]
        status = f"✅ ⬇️  {n} commits"
        return status

    return "??? - master parse fail"


def parseGitOutput(gitOutput, path):
    #  Should maybe switch to using --porcelain for the initial status stuff

    branch = gitOutput.split('\n')[0].split()[2]  # always the third word of first line?
    line3 = gitOutput.split('\n')[2]

    if branch == 'master':
        status = fetchAndCheckMaster(path)
        return branch, status

    if line3.endswith('respectively.'):
        return branch, GITMAP['diverged']

    try:
        status = GITMAP[line3]
    except KeyError:
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
    if len(args) > 1:
        sorting = args[1]
        sorting = sorting.replace('-', '')
        sorting = sorting[0]
        if sorting == '?':
            dumpGITMAP()
            exit()
        assert sorting in ['p', 's', 't']

    cmd = 'eups list -s'
    eupsOutput = subprocess.check_output(cmd.split(), universal_newlines=True)

    packages, paths = getLocalPackagesFromEupsOutput(eupsOutput)
    branches, statuses = [], []

    for package, path in zip(packages, paths):
        branch, status = getBranchAndStatus(package, path)
        branches.append(branch)
        statuses.append(status)

    assert len(packages) == len(paths) == len(branches) == len(statuses)
    print_results(packages, paths, branches, statuses, sorting)
