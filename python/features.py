#!/usr/bin/env python
import sys,os,glob
sys.path.insert(0, '/Users/jui-hsien/code/pyAudioAnalysis')
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
from matplotlib import cm

from util import LoadFeatures

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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
        return F.reshape(F.shape[0]*F.shape[1])
    return extract

def ComputeFeaturesEmbedding(features):
    pca = PCA(n_components=50)
    features_embedded = pca.fit_transform(features)
    features_embedded = TSNE(n_components=2, verbose=1).fit_transform(features_embedded)
    print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)
    # features_embedded = TSNE(n_components=2, learning_rate=1000.0, verbose=1).fit_transform(features)
    return features_embedded

if __name__ == '__main__':
    # visualize the features embedding
    materials = ['ceramics', 'glass', 'steel', 'wood', 'abs', 'polycarbonate']
    objects = ['mug', 'wine', 'ruler']
    colors = [np.random.rand(1,3) for x in np.linspace(0, 1, len(materials)*len(objects))]
    plt.figure(figsize=[12,10])
    X = None
    C = None
    L = [] # labels
    # use_subset=range(0,3) # set 1
    # use_subset=range(3,8) # set 2
    # use_subset=range(8,21) # set 3
    # use_subset=range(21,34) # set 4
    use_subset=range(3,34) # set 5
    # use_subset=None # set 6
    offset = [0]
    for jj, obj in enumerate(objects):
        for ii, material in enumerate(materials):
            files = glob.glob('training*%s*%s*features_bin.npy' %(obj,material))
            for f in files:
                print 'loading features', f
                features = LoadFeatures(f, binary=True)
                if X is None:
                    X = features
                    C = np.repeat(colors[ii+jj*len(materials)], features.shape[0], axis=0)
                else:
                    X = np.concatenate((X, features))
                    C = np.concatenate((C, np.repeat(colors[ii+jj*len(materials)], features.shape[0], axis=0)))
                offset.append(offset[len(offset)-1]+features.shape[0])
                L.append('(%s,%s)' %(obj, material))
    embed = ComputeFeaturesEmbedding(X)
    # plt.plot(embed[:,0], embed[:,1], 'o', color=colors[ii])
    for ii in range(len(offset)-1):
        plt.scatter(embed[offset[ii]:offset[ii+1],0],
                    embed[offset[ii]:offset[ii+1],1],
                    color=C[offset[ii]:offset[ii+1],:], label=L[ii])
    plt.legend()
    plt.savefig('embedding_set5-features.pdf')
    plt.show()
