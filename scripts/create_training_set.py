#!/usr/bin/env python
from util import *
from features import ExtractorAudioAnalysis
import glob,os

MATERIALS_ROOT = '/Users/jui-hsien/code/modal_classify/data/materials/'
OBJECTS_ROOT = '/Users/jui-hsien/code/modal_classify/data/objects/'

out = 'dataset3'
out_prefix = 'training-set'
obj = 'ruler'
sample_secs = 1.
N_samples = 1000
materials = ['ceramics', 'glass', 'steel', 'wood', 'abs', 'polycarbonate']
overwrite = 0

RunCmd('mkdir -p %s' %(out), exe=True)

required_files = []
sim_parameters = []
for mat in materials:
    file_out = '%s/%s_%s_%s.dat' %(out, out_prefix, obj, mat)
    file_mat = MATERIALS_ROOT + mat + '.txt'
    dir_out = '%s/%s_%s_%s' %(out, out_prefix, obj, mat)
    required_files.append(file_mat)
    sim = SimParameters(file_mat, file_out, N_samples, sample_secs, dir_out)
    sim_parameters.append(sim)

file_obj = OBJECTS_ROOT + obj + '/%s.tet.obj' %(obj)
file_mod = OBJECTS_ROOT + obj + '/%s_surf.modes' %(obj)
required_files.append(file_obj)
required_files.append(file_mod)

# run simulations to get training data
print 'Running simulations'
if CheckFilesExist(required_files):
    Run_Sims(file_obj, file_mod, sim_parameters, overwrite=overwrite)
else:
    print '**ERROR** Missing one of the following files, sim not run'
    print '  {}'.format(required_files)

# extract features for each sims
print 'Extracting features'
feature_extractor = ExtractorAudioAnalysis()
for sim in sim_parameters:
    RunCmd('mkdir -p %s' %(sim.dir_out), exe=True)
    data = Read_Training_Set(sim.file_out)
    sim.file_features = sim.file_out.replace('.dat', '.features')
    sim.file_vids = sim.file_out.replace('.dat', '.vids')
    print '  computing features for : {}'.format(sim.file_features)
    if not os.path.isfile(sim.file_features):
        wavfile_list = Write_Wavs(data, sim.dir_out)
        features = ComputeFeatures(wavfile_list, feature_extractor)
        SaveFeatures(features, sim.file_features, overwrite=overwrite)
    if not os.path.isfile(sim.file_vids):
        with open(sim.file_vids, 'w') as stream:
            for d in data:
                stream.write('%u\n' %d[0])

