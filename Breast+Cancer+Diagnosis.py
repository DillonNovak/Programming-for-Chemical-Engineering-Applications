
# coding: utf-8

# ## Predicting Malignant Tumors
# ### Wisconsin Diagnostic Beast Cancer Dataset
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# Dataset attributes:
#      
#      0. diagnosis (malignant or benign)
#      
#      1. radius (mean of distances from center to points on the perimeter)
# 	 2. texture (standard deviation of gray-scale values)
# 	 3. perimeter
# 	 4. area
# 	 5. smoothness (local variation in radius lengths)
# 	 6. compactness (perimeter^2 / area - 1.0)
# 	 7. concavity (severity of concave portions of the contour)
# 	 8. concave points (number of concave portions of the contour)
# 	 9. symmetry 
# 	 10. fractal dimension ("coastline approximation" - 1)

# In[20]:

get_ipython().magic('matplotlib inline')
from sklearn.decomposition import PCA
import sys
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import seaborn as sns
sns.set_context('talk')


# In[21]:

#import PCA models

from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[22]:

# load dataset

dfall = pd.read_csv('wdbc.data.txt')
# drop standard error and largest value for each attribute
df = dfall.drop(dfall.columns[[0,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31]],axis=1)
# name columns
df.columns = ['diagnosis','radius','texture','perimeter','area','smoothness','comactness','concavity','concave points','symmetry','fractal dimension']


# In[23]:

print(df.shape)
print(df.describe())


# In[24]:

print(df.groupby('diagnosis').size())
df.head()


# In[25]:

scatter_matrix(df)
plt.show()


# In[26]:

X = df.ix[:,1:11]
X.tail()


# In[30]:

# plot histogram distribution of each attribute
plt.figure(figsize=(16,6))
plt.subplot(2,5,1)
k = 1
for c in X.columns:
    plt.subplot(2,5,k)
    plt.hist(X[c],normed=True,alpha=0.6,bins=20)
    plt.title(c)
    k += 1
plt.tight_layout()


# ### Scaling and Centering

# In[32]:

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

lbls = X.columns

plt.figure(figsize=(16,6))
plt.subplot(2,5,1)
k = 0
for c in lbls:
    plt.subplot(2,5,k+1)
    plt.hist(X_std[:,k],normed=True,alpha=0.6,bins=20)
    plt.title(c)
    k += 1
plt.tight_layout()


# ### PCA Analysis

# In[33]:

pca = PCA(n_components=3)
Y = pca.fit_transform(X_std)


# In[34]:

w = pca.components_
v = pca.explained_variance_ratio_
print(v)

for k in range(0,len(w)):
    plt.subplot(3,1,k+1)
    plt.bar(range(0,len(w[k])),w[k],width=.5)
    plt.xticks(range(0,len(w[k])),lbls)
    plt.title('explained variance ratio = {0:.3f}'.format(v[k]))

plt.tight_layout()


# In[35]:

k = 0
for n in df['diagnosis']:
    if(df.ix[k,0] == 'M'):
        plt.scatter(Y[k,0],Y[k,1],color='red',alpha=0.4)
    else:
        plt.scatter(Y[k,0],Y[k,1],color='green',alpha=0.4)
    k += 1


# ### Predictive Analysis
# Train predictive models to identify malignant tumors and choose the most accurate model to test on a validation set of data

# In[43]:

# split out validation dataset
diagnosisarray = df.values
dfvals = df.drop(df.columns[[0]],axis=1)
#print(dfvals.head())
array = dfvals.values
X = array[:,0:9]
Y = diagnosisarray[:,0]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[44]:

#check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#evaluate each model
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[45]:

# compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# LDA is the most accurate.
# Use LDA model to evaluate the validation dataset

# In[16]:

# make predictions on validation dataset
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[47]:

# make predictions on validation dataset
lr = LogisticRegression()
lr.fit(X_train, Y_train)
predictions = lr.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

