import subprocess
import os
import shutil
from data_loading import cd

if __name__ == "__main__":
    path = "C:/SOFTWARE and giggles/HORIISON/Other projects/IlluminatingSingularity"
    dst_folder = "C:/Users/jeppe/Dropbox/MI/uno"

    while True:
        # Opening log destination files
        std_out1 = open("Output_log.txt", "w")
        std_err1 = open("Error_log.txt", "w")

        # Running subprocess and logging output and errors
        with cd(path):
            subprocess.run(['python', 'main.py'], stderr=std_err1)

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