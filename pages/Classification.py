#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st


st.sidebar.header("AirBnB data")
st.title('Classification for low or high prices')
st.markdown('The features were chosen based in relevancy and correlation to the target price. The following features utilized were: \n - Minimum nights \n - Availability over the year \n - Room type \n - Neighbourhood \n  - Longitude \n - Latidude \n - The test set was 30% of the data and the remaining 70% was the training data.')
data = pd.read_csv('Listings_clean.csv')

# function to evaluate predictions
def evaluate(y_true, y_pred, classifier):
    # calculate and display confusion matrix
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
    print(cm)

    # calculate precision, recall, and F1 score
    accuracy = float(np.trace(cm)) / np.sum(cm)
    if classifier == 'binary':
      precision = precision_score(y_true, y_pred, average='binary', labels=labels)
      recall = recall_score(y_true, y_pred, average='binary' , labels=labels)
      f1 = f1_score(y_true,y_pred, average = 'binary')
    else:
      precision = precision_score(y_true, y_pred, average='weighted', labels=labels)
      recall = recall_score(y_true, y_pred, average='weighted' , labels=labels)
      f1 = 2 * precision * recall / (precision + recall)

    st.write("accuracy:", accuracy)
    st.write("precision:", precision)
    st.write("recall:", recall)
    st.write("f1 score:", f1)

    #Making a target variable based on price (the mean is the threshold).
price_threshold = data['price'].median()
data['rental_price_binary'] = data['price'].apply(lambda x: 1 if x > price_threshold else 0)

features = data[['minimum_nights', 'availability_365', 'room_type','neighbourhood', 'longitude', 'latitude']]

target = data['rental_price_binary']

#Splitting to test and train
split=int(len(data)*0.7)
x_train = features[:split]
x_test = features[split:]
y_train = target[:split]
y_test = target[split:]

#Making dummy variables
x_train = pd.get_dummies(x_train, columns=['neighbourhood', 'room_type'])
x_test = pd.get_dummies(x_test, columns=['neighbourhood', 'room_type'])


#Standardizing the data
x_train_std = x_train.copy()
x_test_std = x_test.copy()

features_std =['minimum_nights', 'availability_365','longitude', 'latitude']

x_train_std[features_std] = (x_train[features_std] - x_train[features_std].mean()) / x_train[features_std].std()
x_test_std[features_std] = (x_test[features_std] - x_train[features_std].mean()) / x_train[features_std].std()

model = st.selectbox(
    'Pick a classification model',
    ('Logistic Regression', 'Random Forrest', 'Support Vector Machine'))


#Making a logistic regression binary classifier
if model == 'Logistic Regression':
  lr = LogisticRegression(max_iter = 10000)
  lr.fit(x_train_std,y_train)
  y_pred = lr.predict(x_test_std)
elif model == 'Random Forrest':
  clf = RandomForestClassifier(max_depth=2, random_state=0)
  clf.fit(x_train_std,y_train)
  y_pred=clf.predict(x_test_std)
elif model == 'Support Vector Machine': 
  svc =SVC()
  svc.fit(x_train_std, y_train)
  y_pred = svc.predict(x_test_std)
   

# make predictions from Logistic regression model





code = '''    #Making a target variable based on price (the mean is the threshold).
price_threshold = data['price'].median()
data['rental_price_binary'] = data['price'].apply(lambda x: 1 if x > price_threshold else 0)

features = data[['minimum_nights', 'availability_365', 'room_type','neighbourhood', 'longitude', 'latitude']]

target = data['rental_price_binary']

#Splitting to test and train
split=int(len(data)*0.7)
x_train = features[:split]
x_test = features[split:]
y_train = target[:split]
y_test = target[split:]

#Making dummy variables
x_train = pd.get_dummies(x_train, columns=['neighbourhood', 'room_type'])
x_test = pd.get_dummies(x_test, columns=['neighbourhood', 'room_type'])


#Standardizing the data
x_train_std = x_train.copy()
x_test_std = x_test.copy()

features_std =['minimum_nights', 'availability_365','longitude', 'latitude']

x_train_std[features_std] = (x_train[features_std] - x_train[features_std].mean()) / x_train[features_std].std()
x_test_std[features_std] = (x_test[features_std] - x_train[features_std].mean()) / x_train[features_std].std()


#Making a logistic regression binary classifier
lr = LogisticRegression(max_iter = 10000)
lr.fit(x_train_std,y_train)

# make predictions from Logistic regression model
y_pred_lr = lr.predict(x_test_std)'''



# Evaluation the binary classifier
st.subheader('Evaluation from the model')
evaluate(y_test, y_pred, 'binary')




labels = np.unique(y_test)

# Create a heatmap from the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       xlabel='Predicted label', ylabel='True label',
       title='Confusion matrix',
       aspect='equal')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
st.pyplot(fig)

if st.checkbox('Show binary prediction code'):
    st.subheader('binary prediction code')
    st.code(code, language='python')

if st.checkbox('Show coefficient from the model'):
    st.subheader('Coefficients:')
    for colname, val in zip(x_train.columns, lr.coef_.tolist()[0]):
        st.write("%s=%.3f"%(colname, val))
