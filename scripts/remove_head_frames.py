#!/usr/bin/env python
import glob,os,sys,subprocess
def RunCmd(cmd, exe=True):
    print cmd
    if exe:
        subprocess.call(cmd, shell=True)
filenames = sorted(glob.glob('test-*.png'), key=lambda x:
                   int(x.split('-')[-1].split('.')[0]))
print filenames
start_from = int(sys.argv[1])
RunCmd('mkdir tmp')
count = 0
for ii in range(start_from, start_from+1800):
    RunCmd('cp %s tmp/test-%0.4u.png' %(filenames[ii], count))
    count += 1
RunCmd('ffmpeg -r 30 -i tmp/test-%4d.png -c:v libx264 -r 30 -crf 5 -qcomp 1.0 -pix_fmt yuv420p movie.mp4')
# RunCmd('rm -r tmp')
