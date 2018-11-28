import glob
import numpy as np
from util import LoadFeatures

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

def TrainWithLinearSVM(features_folder, train_test_split=0.5, use_subset=None):
    file_features_list = glob.glob('%s/training-set*.features' %(features_folder))
    train_X = None
    train_Y = []
    test_X = None
    test_Y = []
    print 'Training with linear svm'
    print '  Gathering data'
    for idx, file_features in enumerate(file_features_list):
        if use_subset is not None:
            print 'use_subset', use_subset
            features = LoadFeatures(file_features, use_subset=use_subset)
        else:
            features = LoadFeatures(file_features)
        split_bound = int(train_test_split*features.shape[0])
        if idx == 0:
            train_X = features[:split_bound, :]
            test_X = features[split_bound:, :]
        else:
            train_X = np.concatenate((train_X, features[:split_bound, :]))
            test_X = np.concatenate((test_X, features[split_bound:, :]))
        train_Y.extend(['%s' %(file_features) for _ in range(len(range(split_bound)))])
        test_Y.extend(['%s' %(file_features) for _ in range(features.shape[0] - len(range(split_bound)))])

    print '  Training classifier'
    classifier = LinearSVC(tol=1E-3, max_iter=1000, dual=False)
    classifier.fit(train_X, train_Y)

    print '  Test classifier'
    predict_Y = classifier.predict(test_X)
    wrong = 0
    for ii in range(len(predict_Y)):
        if predict_Y[ii] != test_Y[ii]:
            print 'wrong prediction -> predict : valid = ', predict_Y[ii], test_Y[ii]
            wrong += 1
    print '    wrong ', wrong
    print '    |validation| ', len(predict_Y)
    print '    wrong_percentage ', float(wrong)/len(predict_Y)
    print '    accuracy ', 1-float(wrong)/len(predict_Y)
    print confusion_matrix(predict_Y, test_Y)
    print classification_report(test_Y, predict_Y)


if __name__ == '__main__':
    # TrainWithLinearSVM('.', use_subset=range(0,3)) # set 1
    # TrainWithLinearSVM('.', use_subset=range(3,8)) # set 2
    # TrainWithLinearSVM('.', use_subset=range(8,21)) # set 3
    # TrainWithLinearSVM('.', use_subset=range(21,34)) # set 4
    # TrainWithLinearSVM('.', use_subset=range(3,34)) # set 5
    TrainWithLinearSVM('.', use_subset=None) # set 6
