import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

#Import Dataset
dataset = pd.read_csv('dataset6.csv', delimiter=",")

# Split into matrix of features and Matrix of Dependent Variables
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encode categorical varaibles
label_encoder_X = LabelEncoder()
#X[:,1] = label_encoder_X.fit_transform(X[:,1])
label_encoder_y = LabelEncoder()
#y = label_encoder_y.fit_transform(y)

# Scale Data

st_sc = StandardScaler()

print(st_sc.fit(X))

X = st_sc.fit_transform(X)


#print(y)

# Split X and y into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 0)







# Create Model
model = Sequential()

#model.add(Dense(20, input_dim = 10, activation = 'relu'))
#model.add(Dense(80, activation = 'relu'))
#model.add(Dense(130, activation = 'relu'))
#model.add(Dense(80, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(20, activation = 'relu'))
#model.add(Dense(1, activation = 'sigmoid'))
model.add(Dense(15, input_dim = 10, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 200, epochs = 1)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred = model.predict(X)
print(y_pred)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y, y_pred)
print(cm)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#print(X_test)

print(model.predict(np.array([[43, 0, 0.8, 0.2, 192, 29, 20, 6, 2.9, 0.9]])))

#model.save('my_model.h5')
del model
model = load_model('my_model.h5')


print(model.predict(np.array(st_sc.transform([[30, 1, 2, 0.1, 99, 40, 20, 20, 1, 0.5]]))))
print(model.predict(np.array(st_sc.transform([[30, 1, 2, 0.1, 99, 40, 20, 20, 1, 1]]))))
print(model.predict(np.array(st_sc.transform([[4, 0, 0.9, 0.2, 348, 30, 34, 8, 4, 1]]))))

model = load_model('his_model.h5')


print(model.predict(np.array(st_sc.transform([[30, 1, 2, 0.1, 99, 40, 20, 20, 1, 0.5]]))))
print(model.predict(np.array(st_sc.transform([[30, 1, 2, 0.1, 99, 40, 20, 20, 1, 1]]))))
print(model.predict(np.array(st_sc.transform([[4, 0, 0.9, 0.2, 348, 30, 34, 8, 4, 1]]))))
