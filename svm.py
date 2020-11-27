import pandas as pd
import matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC


data_set = pd.read_csv('/Users/wguo/GitRepo/adml/datasets/pima-indians-diabetes.csv')
data = data_set.values[:,:]
'''
plt.figure(figsize=(16,12))
sns.heatmap(data_set.corr(),annot=True,fmt=".2f")
plt.show()
'''
y = data[:,8]
X = data[:,:8]
print(X)
print(y)
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Similar to SVC with parameter kernel=’linear’, but implemented
# in terms of liblinear rather than libsvm, so it has more flexibility
# in the choice of penalties and loss functions and should scale better
# to large numbers of samples.
clf = Pipeline([
    ('standardscaler', StandardScaler()),
    ('linearsvc', LinearSVC(C=1.0, random_state=0, tol=1e-05))
])
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("LinearSVC")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)

## how a classifier is optimized by cross-validation, which is done using the
# sklearn.model_selection.GridSearchCV object on a development set 
# that comprises only half of the available labeled data.
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 2, 3]},
    {'kernel': ['linear'], 'C': [1, 2, 3]}
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(
        SVC(), tuned_parameters, cv=5
    )
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
