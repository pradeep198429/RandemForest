import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("Fraud_check.csv")

df=df.rename(columns={'Taxable.Income':'Tax','Marital.Status':'Status','City.Population':'Population','Work.Experience':'Exp'})

df.loc[df.Tax<=30000,'Tax']='Risk'
df.loc[df.Tax != 'Risk','Tax'] ='Good'

df=df.iloc[:,[2,0,1,3,4,5]]
df=pd.get_dummies(df,columns=['Undergrad','Status','Urban'],drop_first=True)

train,test=train_test_split(df,test_size=0.3)

model=RandomForestClassifier(n_estimators=100)
model.fit(train.iloc[:,1:7],train.iloc[:,0])

#Train test accuracy

train_acc=np.mean(model.predict(train.iloc[:,1:7])==train.iloc[:,0])
test_acc=np.mean(model.predict(test.iloc[:,1:7])==test.iloc[:,0])

acc=[]
for i in range(100,200,2):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train.iloc[:,1:7],train.iloc[:,0])
    train_acc = np.mean(clf.predict(train.iloc[:,1:7])==train.iloc[:,0])
    test_acc = np.mean(clf.predict(test.iloc[:,1:7])==test.iloc[:,0])
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations

# train accuracy plot
plt.plot(np.arange(100,200,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(100,200,2),[i[1] for i in acc],"bo-")


plt.legend(["train","test"])

####################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("Fraud_check.csv")

df=df.rename(columns={'Taxable.Income':'Tax','Marital.Status':'Status','City.Population':'Population','Work.Experience':'Exp'})

df.loc[df.Tax<=30000,'Tax']='Risk'
df.loc[df.Tax != 'Risk','Tax'] ='Good'

df=df.iloc[:,[2,0,1,3,4,5]]
df=pd.get_dummies(df,columns=['Undergrad','Status','Urban'],drop_first=True)

X=df.iloc[:,1:7]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

classification_report(y_test,y_pred)










