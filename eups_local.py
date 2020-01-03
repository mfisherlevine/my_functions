import subprocess
import numpy as np
import sys

GITMAP = {
    "nothing to commit, working tree clean": "✅",
    "Untracked files:": "🤷‍♂️",
    "Changes not staged for commit:": "❌",
    "Parsing failed": "⛔️",
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

    for package, path, branch, status in zip(packages[inds], paths[inds], branches[inds], statuses[inds]):
        print(f"{package:15}\t{path:30}\t{branch:18}\t{status}")


def parseGitOutput(gitOutput):
    branch = gitOutput.split('\n')[0].split()[2]
    l3 = gitOutput.split('\n')[2]
    try:
        status = GITMAP[l3]
    except KeyError:
        print(f"Failed to map the following git status for {branch}:")
        print('---------')
        show(gitOutput)
        print('---------')
        status = "⛔️"
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

    lines = [line for line in eupsOutput.split('\n') if "LOCAL" in line]
    packages, paths = [], []
    branches, statuses = [], []
    for line in lines:
        ret = line.split()
        assert len(ret) == 3
        packages.append(ret[0])
        paths.append(ret[1][6:])

    for package, path in zip(packages, paths):
        cmd = f"git --git-dir={path}/.git --work-tree={path} status"
        gitOutput = subprocess.check_output(cmd.split(), universal_newlines=True)
        branch, status = parseGitOutput(gitOutput)
        branches.append(branch)
        statuses.append(status)

    assert len(packages) == len(paths) == len(branches) == len(statuses)
    print_results(packages, paths, branches, statuses, sorting)
