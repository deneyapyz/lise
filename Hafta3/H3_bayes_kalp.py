import numpy as np
import pandas as pd

train =pd.read_csv("mitbih_train.csv")
X_train=np.array(train)[:,:187]
y_train=np.array(train)[:,187]


test =pd.read_csv("mitbih_test.csv")
X_test=np.array(test)[:,:187]
y_test=np.array(test)[:,187]


from sklearn.naive_bayes import CategoricalNB
gnb = CategoricalNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm= confusion_matrix(y_test,y_pred)
index = ['No','S','V','F','Q'] 
columns = ['No','S','V','F','Q']   
cm_df = pd.DataFrame(cm,columns,index)                      
plt.figure(figsize=(10,6))  
sns.heatmap(cm_df, annot=True,fmt="d",cmap="YlGnBu")

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
