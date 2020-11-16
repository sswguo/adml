#https://www.cnblogs.com/pythonbao/p/10859517.html#_label2
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

data_set = pd.read_csv('/Users/wguo/GitRepo/adml/pima-indians-diabetes.csv')
data = data_set.values[:,:]

y = data[:,8]
X = data[:,:8]
X_train,X_test,y_train,y_test = train_test_split(X,y)

### 随机森林
print("==========================================")   
RF = RandomForestClassifier(n_estimators=10,random_state=11)
RF.fit(X_train,y_train)
predictions = RF.predict(X_test)
print("RF")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))


### Logistic Regression Classifier 
print("==========================================")      
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("LR")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))
 
 
### Decision Tree Classifier    
print("==========================================")   
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("DT")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

 
### GBDT(Gradient Boosting Decision Tree) Classifier    
print("==========================================")   
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("GBDT")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

 
###AdaBoost Classifier
print("==========================================")   
from sklearn.ensemble import  AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("AdaBoost")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

 
### GaussianNB
print("==========================================")   
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("GaussianNB")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

 
### Linear Discriminant Analysis
print("==========================================")   
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("Linear Discriminant Analysis")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))

 
### Quadratic Discriminant Analysis
print("==========================================")   
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("Quadratic Discriminant Analysis")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))


### SVM Classifier 
print("==========================================")   
from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("SVM")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))


### Multinomial Naive Bayes Classifier
print("==========================================")       
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.01)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("Multinomial Naive Bayes")
print(classification_report(y_test,predictions))
print("AC",accuracy_score(y_test,predictions))


### xgboost
#import xgboost
#print("==========================================")       
#from sklearn.naive_bayes import MultinomialNB
#clf = xgboost.XGBClassifier()
#clf.fit(X_train,y_train)
#predictions = clf.predict(X_test)
#print("xgboost")
#print(classification_report(y_test,predictions))
#print("AC",accuracy_score(y_test,predictions))


### voting_classify
#from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
#import xgboost
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#clf1 = GradientBoostingClassifier(n_estimators=200)
#clf2 = RandomForestClassifier(random_state=0, n_estimators=500)
# clf3 = LogisticRegression(random_state=1)
# clf4 = GaussianNB()
#clf5 = xgboost.XGBClassifier()
#clf = VotingClassifier(estimators=[
    # ('gbdt',clf1),
#    ('rf',clf2),
    # ('lr',clf3),
    # ('nb',clf4),
    # ('xgboost',clf5),
 #   ],
 #   voting='soft')
#clf.fit(X_train,y_train)
#predictions = clf.predict(X_test)
#print("voting_classify")
#print(classification_report(y_test,predictions))
#print("AC",accuracy_score(y_test,predictions))