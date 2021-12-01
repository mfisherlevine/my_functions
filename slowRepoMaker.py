import glob
import os.path
from time import sleep
import shutil
import subprocess


class NewImages():
    pauseLength = 30

    def __init__(self, repoDir, dataGlob, clobber=False):
        self.repoDir = repoDir
        self.clobber = clobber
        self.files = sorted([os.path.abspath(f) for f in glob.glob(dataGlob)])

    def _preRun(self):
        if os.path.exists(self.repoDir):
            if self.clobber:
                shutil.rmtree(repoDir)
                os.makedirs(repoDir)
                calibDir = os.path.join(repoDir, 'CALIB')
                os.makedirs(calibDir)
                shutil.copy('/project/shared/auxTel/CALIB/calibRegistry.sqlite', calibDir)
                shutil.copy('/project/shared/auxTel/CALIB/calibRegistry.sqlite3', calibDir)
                shutil.copy('/project/shared/auxTel/CALIB/repositoryCfg.yaml', calibDir)
            else:
                raise RuntimeError(f"Path {repoDir} already exists - please use clobber=True to recreate")
        else:
            os.makedirs(repoDir)

        with open(os.path.join(repoDir, "_mapper"), "w") as f:
            f.write("lsst.obs.lsst.auxTel.AuxTelMapper")

    def run(self):
        self._preRun()
        for filename in self.files:
            args = f" ingestImages.py {repoDir} {filename} --mode=link"
            print(args)
            ret = subprocess.check_output(args.split(), universal_newlines=True)
            sleep(self.pauseLength)


if __name__ == "__main__":
    dataGlob = '/project/shared/auxTel/_parent/raw/2021-06-08/20210608004*.fits'
    repoDir = '/home/mfl/newImageRepo/'
    repoMaker = NewImages(repoDir, dataGlob, clobber=True)
    repoMaker.run()
