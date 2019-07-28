#!/usr/bin/env python
import subprocess
outroot = '/Users/jui-hsien/code/modal_demo/data/10k_2'
def RunCmd(cmd, exe=True):
    print cmd
    if exe:
        subprocess.call(cmd, shell=True)
with open('ran_obj_mat.txt', 'r') as stream:
    lines = stream.readlines()
    for l in lines:
        tokens = l.split()
        ID = tokens[0].split('/')[-1]
        path= tokens[0]
        mat = tokens[1]
        outdir = '%s/%s_tetmesh' %(outroot, ID)
        # cmd='mkdir -p %s' %(outdir)
        # RunCmd(cmd)
        # cmd='scp monopole:%s/%s_tetmesh.tet.obj %s/' %(path, ID, outdir)
        # RunCmd(cmd)
        # cmd='scp monopole:%s/modal_models/%s/%s_tetmesh_surf.modes %s/' \
        #         %(path, mat, ID, outdir)
        # RunCmd(cmd)
        # cmd='scp -r monopole:%s/radiation_models/%s/ffat_map-fdtd %s/' \
        #         %(path, mat, outdir)
        # RunCmd(cmd)
        cmd='cp %s/../materials/%s.txt %s' %(outroot, mat, outdir)
        RunCmd(cmd, exe=True)
