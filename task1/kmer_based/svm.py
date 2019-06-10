from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

from load_data import load_data


k = 6
stride = 1

CV = 5
n_jobs = 8

print("loading kmer counts")
X, y = load_data(k, stride)

print("kmer counts loaded")
print(X.shape, y.shape)

clf = SVC(kernel='rbf', C=1.0)

scoring = ['precision', 'recall', 'f1', 'accuracy', 'roc_auc']

# (0.9475181001513816, 0.020155612502537647)
# (0.9290000728279078, 0.034944652489570266)
# (0.9377324592508763, 0.020656615710573128)
# (0.9383987055754144, 0.01996373406242249)
# (0.9835348303528935, 0.007063968897701087)
scores = cross_validate(clf, X, y, cv=CV, scoring=scoring, n_jobs=n_jobs)
for s in scoring:
    s_ = scores["test_" + s]
    print(s_.mean(), s_.std())


