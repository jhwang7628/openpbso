#/usr/bin/env python
from util import ConvertFeaturesBinary,LoadFeatures,BinaryFilename
import glob
import numpy as np

if __name__ == '__main__':
    filenames = sorted(glob.glob('training-set*features'))
    for f in filenames:
        print 'converting features file: ', f
        ConvertFeaturesBinary(f, BinaryFilename(f, False))
        features_1 = LoadFeatures(f)
        features_2 = LoadFeatures(BinaryFilename(f, True), binary=True)
        print 'err = ', np.max(np.abs(features_1 - features_2))

