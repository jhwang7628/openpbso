#!/usr/bin/env python
import sys,os,glob
sys.path.insert(0, '/Users/jui-hsien/code/pyAudioAnalysis')
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt

from util import LoadFeatures

from sklearn.manifold import TSNE

def TestPyAudioAnalysis():
    [Fs, x] = audioBasicIO.readAudioFile("/Users/jui-hsien/code/modal_classify/data/dataset/wine_ceramics/output/71.wav");
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
    plt.subplot(2,1,1);
    plt.plot(F[0,:]);
    plt.xlabel('Frame no');
    plt.ylabel(f_names[0]);
    plt.subplot(2,1,2);
    plt.plot(F[1,:]);
    plt.xlabel('Frame no');
    plt.ylabel(f_names[1]);
    plt.show()

def ExtractorAudioAnalysis(frame_size=0.05, frame_step=0.025):
    def extract(wavfile):
        assert os.path.isfile(wavfile)
        [Fs, x] = audioBasicIO.readAudioFile(wavfile)
        F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_step*Fs);
        print 'F =', np.array(F)
        print F.shape
        print 'F reshape =', F.reshape(F.shape[0]*F.shape[1])
        return F.reshape(F.shape[0]*F.shape[1])
    return extract

def VisualizeFeatures(features):
    features = np.transpose(features)
    features_embedded = TSNE(n_components=2).fit_transform(features)
    print features_embedded.shape

if __name__ == '__main__':
    files = glob.glob('training*features')
    X = None
    for f in files:
        features = LoadFeatures(f)[:,:100]
        if X is None:
            X = features
        else:
            X = np.concatenate((X, features))
    VisualizeFeatures(X)
