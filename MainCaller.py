import subprocess
import os
import shutil

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


if __name__ == "__main__":
    path = "D:/HORIISON/internal_projects/IlluminatingSingularity"
    dst_folder = "D:/HORIISON/internal_projects"

    while True:
        # Opening log destination files
        std_out1 = open("Output_log.txt", "w")
        std_err1 = open("Error_log.txt", "w")

        # Running subprocess and logging output and errors
        with cd(path):
            subprocess.run(['python', 'call_test.py'], stdout=std_out1, stderr=std_err1)

        # closing log files
        std_out1.close()
        std_err1.close()

        # deleting logfiles in destination
        if os.path.isfile(dst_folder+"/Output_log.txt"):
            os.remove(dst_folder+"/Output_log.txt")

        if os.path.isfile(dst_folder + "/Error_log.txt"):
            os.remove(dst_folder + "/Error_log.txt")

        # move log files from subprocess executing path to destination path
        shutil.move(path+"/Output_log.txt", dst=dst_folder+"/Output_log.txt")
        shutil.move(path+"/Error_log.txt", dst=dst_folder+"/Error_log.txt")