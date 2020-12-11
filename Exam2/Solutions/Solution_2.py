#!/usr/bin/env python
# coding: utf-8

# # Solution 2:

# In[252]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.linalg import fractional_matrix_power
from scipy.linalg import sqrtm
from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[253]:


meanVectors = [[-18, -8], [0, 0], [18, 8]]
prior = [0.33, 0.34, 0.33]

N_1 = 1000
N_2 = 10000

M = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
activ = ['sigmoid', 'softplus']


# In[254]:


covEvalues = [[3.2**2, 0], [0, 0.6**2]]
covEvectors_1 = np.multiply(1/math.sqrt(2), [[1, -1], [1, 1]])
covEvectors_2 = [[1, 0], [0, 1]]
covEvectors_3 = np.multiply(1/math.sqrt(2), [[1, -1], [1, 1]])


# In[255]:


#generating number of sample for the GMMs
def sample_number(N):
    l_1 = 0
    l_2 = 0
    l_3 = 0

    for i in range(N):
        temp = np.random.uniform(0, 1, 1)
        if temp <= prior[0]:
            l_1 = l_1 + 1
        elif temp <= prior[0] + prior[1]:
            l_2 = l_2 + 1

    l_3 = N - l_1 - l_2
    return l_1, l_2, l_3


# ### Generating training data:

# In[256]:


train_l_1, train_l_2, train_l_3 = sample_number(N_1)


# In[257]:


train_data = []

train_data_1 = np.add(np.matmul(np.matmul(np.array(covEvectors_1), np.sqrt(np.array(covEvalues))), np.random.randn(2, train_l_1)), np.array(meanVectors)[0, :].reshape((2, 1)))
train_data_2 = np.add(np.matmul(np.matmul(np.array(covEvectors_2), np.sqrt(np.array(covEvalues))), np.random.randn(2, train_l_2)), np.array(meanVectors)[1, :].reshape((2, 1)))
train_data_3 = np.add(np.matmul(np.matmul(np.array(covEvectors_3), np.sqrt(np.array(covEvalues))), np.random.randn(2, train_l_3)), np.array(meanVectors)[2, :].reshape((2, 1)))

for i in range(train_l_1):
    train_data.append(np.array(train_data_1)[:, i])
for i in range(train_l_2):
    train_data.append(np.array(train_data_2)[:, i])
for i in range(train_l_3):
    train_data.append(np.array(train_data_3)[:, i])


# In[258]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(train_data)[:, 0], np.array(train_data)[:, 1], s = 10, label='Data points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Visualization of training data as a 2-D scatter plot')
ax.legend()
plt.show()


# ### Generating testing data:

# In[259]:


test_l_1, test_l_2, test_l_3 = sample_number(N_2)


# In[260]:


test_data = []

test_data_1 = np.add(np.matmul(np.matmul(np.array(covEvectors_1), np.sqrt(np.array(covEvalues))), np.random.randn(2, test_l_1)), np.array(meanVectors)[0, :].reshape((2, 1)))
test_data_2 = np.add(np.matmul(np.matmul(np.array(covEvectors_2), np.sqrt(np.array(covEvalues))), np.random.randn(2, test_l_2)), np.array(meanVectors)[1, :].reshape((2, 1)))
test_data_3 = np.add(np.matmul(np.matmul(np.array(covEvectors_3), np.sqrt(np.array(covEvalues))), np.random.randn(2, test_l_3)), np.array(meanVectors)[2, :].reshape((2, 1)))

for i in range(test_l_1):
    test_data.append(np.array(test_data_1)[:, i])
for i in range(test_l_2):
    test_data.append(np.array(test_data_2)[:, i])
for i in range(test_l_3):
    test_data.append(np.array(test_data_3)[:, i])


# In[261]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(test_data)[:, 0], np.array(test_data)[:, 1], s = 10, label='Data points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Visualization of testing data as a 2-D scatter plot')
ax.legend()
plt.show()


# ### Selecting the optimal model using cross-validation:

# #### Calculating optimal model order for Sigmoid activation

# In[262]:


def baseline_model_1(M=0):
    model = Sequential()
    model.add(Dense(M, input_dim=1, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mse', optimizer='RMSprop', metrics = ['mse'])
    return model


# In[263]:


X = np.array(train_data)[:, 0]
Y = np.array(train_data)[:, 1]
parameters = dict(M=M)
estimator = KerasRegressor(build_fn=baseline_model_1, epochs=1000, batch_size=N_1, verbose=0)
kfold = KFold(n_splits=10, shuffle = True)
gridsrch = GridSearchCV(estimator = estimator, param_grid = parameters, cv = kfold, n_jobs=-1)
result = gridsrch.fit(X, Y)
best_score_1 = -1*result.best_score_
best_param_1 = result.best_params_
mean_1 = np.multiply(-1, result.cv_results_['mean_test_score'])
param_1 = result.cv_results_['params']    


# In[264]:


print("The best parameter for the sigmoid activation function is {} with a loss (mean squared error) of {}".format(best_param_1, best_score_1))


# In[265]:


for mean, param in zip(mean_1, param_1):
    print("%f with: %r" % (mean, param))


# #### Calculating optimal model order for Softplus activation

# In[266]:


def baseline_model_2(M=0):
    model = Sequential()
    model.add(Dense(M, input_dim=1, kernel_initializer='normal', activation='softplus'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics = ['mse'])
    return model


# In[267]:


X = np.array(train_data)[:, 0]
Y = np.array(train_data)[:, 1]
parameters = dict(M=M)
estimator = KerasRegressor(build_fn=baseline_model_2, epochs=1000, batch_size=N_1, verbose=0)
kfold = KFold(n_splits=10, shuffle = True)
gridsrch = GridSearchCV(estimator = estimator, param_grid = parameters, cv = kfold, n_jobs=-1)
result = gridsrch.fit(X, Y)
best_score_2 = -1*result.best_score_
best_param_2 = result.best_params_
mean_2 = np.multiply(-1, result.cv_results_['mean_test_score'])
param_2 = result.cv_results_['params'] 


# In[268]:


print("The best parameter for the softplus activation function is {} with a loss (mean squared error) of {}".format(best_param_2, best_score_2))


# In[269]:


for mean, param in zip(mean_2, param_2):
    print("%f with: %r" % (mean, param))


# In[270]:


if abs(best_score_1) < abs(best_score_2):
    act = 'sigmoid'
    best_param = best_param_1.get('M')
else:
    act = 'softplus'
    best_param = best_param_2.get('M')


# In[271]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(M, mean_1, label='Sigmoid')
ax.plot(M, mean_2, label='Softplus')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Mean Squared Error')
plt.title('Plot of Mean Squared Error vs Number of Perceptrons for both activation functions')
ax.legend()
plt.show()


# ### Predicting the results for the optimal parameters

# In[272]:


X_test = np.array(test_data)[:, 0]
Y_test = np.array(test_data)[:, 1]


# In[273]:


model = Sequential()
model.add(Dense(best_param, input_dim=1, kernel_initializer='normal', activation=act))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mse', optimizer='RMSprop', metrics = ['mse'])
model.fit(X, Y, epochs=1000, batch_size=1000)
loss, accuracy = model.evaluate(X_test, Y_test)
predictions = model.predict(X_test)


# In[274]:


print("The loss (mean squared error) obtained by applying the neural network on the test dataset is {}".format(loss))


# In[275]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(model.history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss (mean squared error)')
plt.title('Plot of loss at various epochs')
plt.show()


# In[276]:


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(test_data)[:, 0], np.array(test_data)[:, 1], s = 10, label='Data points')
ax.scatter(X_test, predictions, s = 10, label = 'Predicted points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of real dataset and predicted points')
ax.legend()
plt.show()


# In[ ]:




