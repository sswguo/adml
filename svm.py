import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif']=[u'simHei']
matplotlib.rcParams['axes.unicode_minus']=False
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

data_set = pd.read_csv('/Users/wguo/GitRepo/adml/datasets/pima-indians-diabetes.csv')
data = data_set.values[:,:]

y = data[:,8]
X = data[:,:8]
X_train,X_test,y_train,y_test = train_test_split(X,y)

### SVM Classifier 
print("==========================================")   
from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("SVM")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

###
clf = SVC(kernel='linear', C=1)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("SVM")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

### https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)

