#!/usr/bin/env python
import numpy as np
import glob
import struct,sys,copy,subprocess,datetime,os
import scipy.io.wavfile
import matplotlib.pyplot as plt

MODAL_CLASSIFY_ROOT = '/Users/jui-hsien/code/modal_classify/'
SIMULATOR = MODAL_CLASSIFY_ROOT + 'build/simulator'

def RunCmd(cmd, exe=True, log=None):
    print cmd
    if log is not None:
        with open(log, 'a') as stream:
            stream.write('{} {}\n'.format(datetime.datetime.now(), cmd))
    if exe:
        subprocess.call(cmd, shell=True)

def CheckFilesExist(files):
    for f in files:
        if not os.path.isfile(f):
            return False
    return True

class SimParameters:
    def __init__(self, file_material, file_out, N_samples, sample_secs, dir_out):
        self.file_material = file_material
        self.file_out = file_out
        self.sample_secs = sample_secs
        self.N_samples = N_samples
        self.dir_out = dir_out

        self.file_features = None
        self.file_vids = None

def Run_Sims(file_obj, file_modes, list_sim_parameters, overwrite=False):
    """ Create a training set consists of different materials """
    for sim in list_sim_parameters:
        if not os.path.isfile(sim.file_out) or overwrite:
            RunCmd('%s -m %s -s %s -t %s -o %s -n %u -l %f' %(SIMULATOR, file_obj,
                                                              file_modes,
                                                              sim.file_material,
                                                              sim.file_out,
                                                              sim.N_samples,
                                                              sim.sample_secs))

def Read_Training_Set(filename, N=-1):
    alldata = []
    with open(filename, 'r') as stream:
        N_samples, N_steps = struct.unpack('ii', stream.read(8))
        if N > 0:
            N_samples = min(N, N_samples)
        for sample in range(N_samples):
            vid = struct.unpack('i', stream.read(4))[0]
            data = np.array(struct.unpack('%uf' %(N_steps), stream.read(N_steps*4)))
            alldata.append((vid, data))
    return alldata

def Write_Wavs(alldata, outdir):
    norm = -1.
    for data in alldata:
        norm = max(norm, max(abs(data[1])))
    filenames = []
    for data in alldata:
        d = np.asarray(data[1]/norm*32767, dtype=np.int16)
        filename = '%s/%u.wav' %(outdir, data[0])
        scipy.io.wavfile.write(filename, 44100., d)
        filenames.append(filename)
    return filenames

def ComputeFeatures(wavfile_list, feature_extractor):
    features = []
    for wavfile in wavfile_list:
        features_i = feature_extractor(wavfile)
        features.append(features_i)
    return np.array(features)

def BinaryFilename(f, extension=False):
    if extension:
        return f + '_bin.npy'
    else:
        return f + '_bin'

def SaveFeatures(features, outfile, overwrite=False):
    if not os.path.isfile(outfile) or overwrite:
        np.savetxt(outfile, features)

def LoadFeatures(filename, use_subset=None, remove_nan_data=True, binary=False):
    if binary:
        features = np.load(filename)
    else:
        features = np.loadtxt(filename)
    if remove_nan_data:
        nanfound = False
        delete_rows = []
        for ii in range(features.shape[0]):
            nanfoundrow = False
            if sum([int(x) for x in np.isnan(features[ii,:])]) > 0:
                nanfoundrow = True
                nanfound = True
            if nanfoundrow:
                delete_rows.append(ii)
        if nanfound:
            features = np.delete(features, delete_rows, 0)
    if use_subset is not None:
        N_features_type = 34
        assert(features.shape[1] % N_features_type == 0)
        N_frames = int(features.shape[1]/N_features_type)
        cols = []
        for s in use_subset:
            assert(s*N_features_type < features.shape[1])
            cols.append(s*N_frames)
        features = features[:, cols]
    return features

def ConvertFeaturesBinary(filename, filename_bin):
    features = np.loadtxt(filename)
    np.save(filename_bin, features)

def TestLoadFeatures():
    filenames = glob.glob('*features_bin.npy')
    for f in filenames:
        features = LoadFeatures(f, binary=True)
        print features.shape
        break

if __name__ == '__main__':
    # data = Read_Training_Set(sys.argv[1], N=100)
    # Write_Wavs(data, 'output')

    TestLoadFeatures()
