import glob
import os.path
from time import sleep
import shutil
import subprocess


class NewImages():
    pauseLength = 10

    def __init__(self, repoDir, dataGlob, clobber=False):
        self.repoDir = repoDir
        self.clobber = clobber
        self.files = sorted([os.path.abspath(f) for f in glob.glob(dataGlob)])

    def _preRun(self):
        if os.path.exists(self.repoDir):
            if self.clobber:
                shutil.rmtree(repoDir)
                os.makedirs(repoDir)
            else:
                raise RuntimeError(f"Path {repoDir} already exists - please use clobber=True to recreate")
        else:
            os.makedirs(repoDir)

        with open(os.path.join(repoDir, "_mapper"), "w") as f:
            f.write("lsst.obs.lsst.auxTel.AuxTelMapper")

    def run(self):
        self._preRun()
        for filename in self.files[0:500]:
            args = f" ingestImages.py {repoDir} {filename} --mode=link"
            print(args)
            ret = subprocess.check_output(args.split(), universal_newlines=True)
            sleep(self.pauseLength)


if __name__ == "__main__":
    dataGlob = '/project/shared/auxTel/_parent/raw/2020-01-28/*.fits'
    repoDir = '/home/mfl/newImageRepo/'
    repoMaker = NewImages(repoDir, dataGlob, clobber=True)
    repoMaker.run()
