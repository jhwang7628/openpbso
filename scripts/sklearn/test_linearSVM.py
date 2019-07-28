from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
print 'X = ', X, X.shape
print 'y = ', y
Y = []
for idx,y_i in enumerate(y):
    if y_i:
        Y.append('a')
    else:
        Y.append('b')

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X, Y)
print clf.coef_
print clf.intercept_
print clf.predict([[0,0,0,0], [1,1,1,1]])
print clf.decision_function([[0,0,0,0], [1,1,1,1]])
