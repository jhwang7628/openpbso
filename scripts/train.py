import glob,time
import numpy as np
from util import LoadFeatures
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score

def SplitTestingSet(X, Y, split_percentage):
    N_total = X.shape[0]
    N_train = int(N_total * split_percentage)
    N_test  = N_total - N_train
    X_train = X[:N_train,:]
    Y_train = Y[:N_train]
    X_test  = X[N_train:N_train+N_test,:]
    Y_test  = Y[N_train:N_train+N_test]
    return X_train, Y_train, X_test, Y_test

def SplitValidationSet(X, Y, v_fold):
    N_total = X.shape[0]
    assert Y.shape[0] == N_total
    # for now discard the undivisible parts
    N_target = int(N_total / v_fold)
    Xi = []
    Yi = []
    for ii in range(v_fold):
        Xi.append(X[ii*N_target:(ii+1)*N_target,:])
        Yi.append(Y[ii*N_target:(ii+1)*N_target])
    return Xi, Yi

def GetY(filename):
    tokens = filename.split('/')[-1].split('.')[0].split('_')
    return tokens[1]+':'+tokens[2] # object - material
    # return tokens[2] # material only

def TrainWithLinearSVM(features_folder, train_test_split=0.9, use_subset=None,
                       C=32768.):
    file_features_list = glob.glob('%s/training-set*.features_bin.npy' %(features_folder))
    train_X = None
    train_Y = []
    test_X = None
    test_Y = []
    print 'Training with linear svm'
    print '  Gathering data'
    print '     use_subset', use_subset
    for idx, file_features in enumerate(file_features_list):
        features = LoadFeatures(file_features, use_subset=use_subset, binary=True)
        split_bound = int(train_test_split*features.shape[0])
        if idx == 0:
            train_X = features[:split_bound, :]
            test_X = features[split_bound:, :]
        else:
            train_X = np.concatenate((train_X, features[:split_bound, :]))
            test_X = np.concatenate((test_X, features[split_bound:, :]))
        train_Y.extend([GetY(file_features) for _ in range(len(range(split_bound)))])
        test_Y.extend([GetY(file_features) for _ in range(features.shape[0] - len(range(split_bound)))])

    print '  Training classifier'
    classifier = LinearSVC(tol=1E-3, max_iter=1000, dual=False, verbose=0, C=C)
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

    return classifier.score(test_X, test_Y), classifier.coef_

def TrainWithLinearSVMCrossValidation(features_folder, train_size=0.9, use_subset=None):
    print 'Training with linear svm using cross-validation'
    print '  Gathering data'
    print '    use_subset', use_subset
    file_features_list = glob.glob('%s/training-set*.features_bin.npy' %(features_folder))
    X = None
    Y = []
    for idx, file_features in enumerate(file_features_list):
        features = LoadFeatures(file_features, use_subset=use_subset, binary=True)
        if idx == 0:
            X = features
        else:
            X = np.concatenate((X, features))
        Y.extend([GetY(file_features) for _ in range(features.shape[0])])

    print '  Splitting test set'
    test_size = (1.0-train_size)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=0)

    Cs = np.logspace(-5, 19, base=2, num=13, endpoint=True)
    print Cs
    bestC = Cs[0]
    bestScore = -1

    # grid search to find the best C
    for C in Cs:
        print '  Training classifier with C = %f' %(C)
        classifier = LinearSVC(tol=1E-3, max_iter=1000, dual=False, verbose=0,
                               C=C)
        scores = cross_val_score(classifier, X_train, Y_train, cv=5)
        print '    score = ', scores
        print "Accuracy: %0.8f (+/- %0.8f)" % (scores.mean(), scores.std() * 2)
        if (scores.mean() > bestScore):
            bestC = C
            bestScore = scores.mean()
    print 'Best C = ', bestC
    print 'Best score = ', bestScore

    # train using the best C
    classifier = LinearSVC(tol=1E-3, max_iter=1000, dual=False, verbose=0, C=bestC)
    classifier.fit(X_train, Y_train)

    print '  Test classifier'
    Y_predict = classifier.predict(X_test)
    wrong = 0
    for ii in range(len(Y_predict)):
        if Y_predict[ii] != Y_test[ii]:
            print 'wrong prediction -> predict : valid = ', Y_predict[ii], Y_test[ii]
            wrong += 1
    print '    wrong ', wrong
    print '    |validation| ', len(Y_predict)
    print '    wrong_percentage ', float(wrong)/len(Y_predict)
    print '    accuracy ', 1-float(wrong)/len(Y_predict)
    print confusion_matrix(Y_predict, Y_test)
    print classification_report(Y_test, Y_predict)

def TestVidsRepeat(filename, train_test_split=0.9):
    vids = np.loadtxt(filename, dtype=int)
    train_vids = set()
    for ii in range(int(len(vids)*train_test_split)):
        train_vids.add(vids[ii])
    for ii in range(int(len(vids)*train_test_split), len(vids)):
        if vids[ii] in train_vids:
            print 'test vid in training set'

def TrainWithSGD(features_folder, train_test_split=0.9, use_subset=None):
    file_features_list = glob.glob('%s/training-set*.features_bin.npy' %(features_folder))
    train_X = None
    train_Y = []
    test_X = None
    test_Y = []
    print 'Training with linear svm using SGD'
    print '  Gathering data'
    print '     use_subset', use_subset
    for idx, file_features in enumerate(file_features_list):
        features = LoadFeatures(file_features, use_subset=use_subset, binary=True)
        split_bound = int(train_test_split*features.shape[0])
        if idx == 0:
            train_X = features[:split_bound, :]
            test_X = features[split_bound:, :]
        else:
            train_X = np.concatenate((train_X, features[:split_bound, :]))
            test_X = np.concatenate((test_X, features[split_bound:, :]))
        train_Y.extend([GetY(file_features) for _ in range(len(range(split_bound)))])
        test_Y.extend([GetY(file_features) for _ in range(features.shape[0] - len(range(split_bound)))])

    print '  Training classifier'
    classifier = linear_model.SGDClassifier(tol=1E-3, max_iter=1000)
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

    return classifier.score(test_X, test_Y), classifier.coef_

def PlotWeights(coeff):
    # coeff_features = np.zeros((coeff.shape[0], 34))
    # for col in range(coeff.shape[1]):
    #     coeff_features[:,col%34] += np.abs(coeff[:,col])
    coeff_features = np.zeros((coeff.shape[0], 5))
    for col in range(coeff.shape[1]):
        if col%34 in range(0,3):
            coeff_features[:,0] += np.abs(coeff[:,col])
        if col%34 in range(3,8):
            coeff_features[:,1] += np.abs(coeff[:,col])
        if col%34 in range(8,21):
            coeff_features[:,2] += np.abs(coeff[:,col])
        if col%34 in range(21,34):
            coeff_features[:,3] += np.abs(coeff[:,col])
        if col%34 in range(3,34):
            coeff_features[:,4] += np.abs(coeff[:,col])
    coeff_features[:,0] /= float(len(range(0,3)))
    coeff_features[:,1] /= float(len(range(3,8)))
    coeff_features[:,2] /= float(len(range(8,21)))
    coeff_features[:,3] /= float(len(range(21,34)))
    coeff_features[:,4] /= float(len(range(3,34)))

    plt.figure()
    for ii in range(coeff_features.shape[0]):
        plt.plot(coeff_features[ii,:], 'o-', label='classifier %u' %(ii))
    # plt.plot(coeff_features[0,:], 'o', label='classifier %u' %(0))
    # plt.legend()
    plt.show()

if __name__ == '__main__':

    # train with cross validation
    # TrainWithLinearSVMCrossValidation('.', use_subset=None) # set 6

    # train with linear svm (liblinear)
    print '\n\nset 1'
    start = time.time()
    TrainWithLinearSVM('.', use_subset=range(0,3)) # set 1
    print '  elapsed time = ', time.time() - start

    print '\n\nset 2'
    start = time.time()
    TrainWithLinearSVM('.', use_subset=range(3,8)) # set 2
    print '  elapsed time = ', time.time() - start

    print '\n\nset 3'
    start = time.time()
    TrainWithLinearSVM('.', use_subset=range(8,21)) # set 3
    print '  elapsed time = ', time.time() - start

    print '\n\nset 4'
    start = time.time()
    TrainWithLinearSVM('.', use_subset=range(21,34)) # set 4
    print '  elapsed time = ', time.time() - start

    print '\n\nset 5'
    start = time.time()
    TrainWithLinearSVM('.', use_subset=range(3,34)) # set 5
    print '  elapsed time = ', time.time() - start

    print '\n\nset 6 (full)'
    start = time.time()
    score, coeff = TrainWithLinearSVM('.', use_subset=None) # set 6
    print '  elapsed time = ', time.time() - start

    # train with linear svm + sgd
    # print '\n\nset 1'
    # start = time.time()
    # TrainWithSGD('.', use_subset=range(0,3)) # set 1
    # print '  elapsed time = ', time.time() - start

    # print '\n\nset 2'
    # start = time.time()
    # TrainWithSGD('.', use_subset=range(3,8)) # set 2
    # print '  elapsed time = ', time.time() - start

    # print '\n\nset 3'
    # start = time.time()
    # TrainWithSGD('.', use_subset=range(8,21)) # set 3
    # print '  elapsed time = ', time.time() - start

    # print '\n\nset 4'
    # start = time.time()
    # TrainWithSGD('.', use_subset=range(21,34)) # set 4
    # print '  elapsed time = ', time.time() - start

    # print '\n\nset 5'
    # start = time.time()
    # TrainWithSGD('.', use_subset=range(3,34)) # set 5
    # print '  elapsed time = ', time.time() - start

    # print '\n\nset 6 (full)'
    # start = time.time()
    # score, coeff = TrainWithSGD('.', use_subset=None) # set 6
    # print '  elapsed time = ', time.time() - start

    # np.savetxt('coeff_sgd', coeff)
    # coeff = np.loadtxt('coeff')
    # PlotWeights(coeff)

    # TestVidsRepeat('training-set_mug_wood.vids')
