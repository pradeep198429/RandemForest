
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("Company_Data.csv")

df = pd.get_dummies(df, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

df.loc[df.Sales <= 10, 'Sales'] = 'No'

df.loc[df.Sales != 'No', 'Sales'] = 'Yes'

train, test = train_test_split(df, test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(train.iloc[:, 1:12], train.iloc[:, 0])

# To find train and test accuracy

train_acc = np.mean(model.predict(train.iloc[:, 1:12]) == train.iloc[:, 0])
test_acc = np.mean(model.predict(test.iloc[:, 1:12]) == test.iloc[:, 0])

acc = []

for i in range(100, 200, 2):
    model = RandomForestClassifier(n_estimators=i)
    model.fit(train.iloc[:, 1:12], train.iloc[:, 0])
    train_acc = np.mean(model.predict(train.iloc[:, 1:12]) == train.iloc[:, 0])
    test_acc = np.mean(model.predict(test.iloc[:, 1:12]) == test.iloc[:, 0])
    acc.append([train_acc, test_acc])

    import matplotlib.pyplot as plt  # library to do visualizations

# train accuracy plot
plt.plot(np.arange(100, 200, 2), [i[0] for i in acc], "ro-")

# test accuracy plot
plt.plot(np.arange(100, 200, 2), [i[1] for i in acc], "bo-")

plt.legend(["train", "test"])
plt.show()

###############################################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("Company_Data.csv")

df = pd.get_dummies(df, columns=['ShelveLoc', 'Urban', 'US'], drop_first=True)

df.loc[df.Sales <= 10, 'Sales'] = 'No'
df.loc[df.Sales != 'No', 'Sales'] = 'Yes'

X = df.iloc[:, 1:12]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)

























