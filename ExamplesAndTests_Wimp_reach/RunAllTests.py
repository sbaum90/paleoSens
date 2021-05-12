import os
import subprocess

fin_list = os.listdir()
os.chdir("../")

for fname in fin_list:
   subprocess.call("python3 WIMP_reach.py ExamplesAndTests_Wimp_reach."+fname[:-3], shell=True)
