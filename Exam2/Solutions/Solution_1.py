#!/usr/bin/env python
# coding: utf-8

# # Solution 1:

# ### Part 1:

# In[170]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# In[171]:


mu = [[2, 2, 0], [-2, 2, 0], [-2, -2, 0], [2, -2, 0]]
var_1 = make_spd_matrix(3)
var_2 = make_spd_matrix(3)
var_3 = make_spd_matrix(3)
var_4 = make_spd_matrix(3)
prior = [0.30, 0.25, 0.35, 0.10]

N_1 = 1000
N_2 = 10000


# In[172]:


#generating number of sample for the GMMs
def sample_number(N):
    l_1 = 0
    l_2 = 0
    l_3 = 0
    l_4 = 0

    for i in range(N):
        temp = np.random.uniform(0, 1, 1)
        if temp <= prior[0]:
            l_1 = l_1 + 1
        elif temp <= prior[0] + prior[1]:
            l_2 = l_2 + 1
        elif temp <= prior[0] + prior[1] + prior[2]:
            l_3 = l_3 + 1

    l_4 = N - l_1 - l_2 - l_3
    return l_1, l_2, l_3, l_4


# In[173]:


#generating data
def data_generator(l_1, l_2, l_3, l_4, mu, var_1, var_2, var_3, var_4):
    data = []
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    true_label = []
    true_prob = []
    N = l_1 + l_2 + l_3 + l_4

    for i in range(l_1):
        temp = np.random.multivariate_normal(mu[0], var_1, 1)
        data_1.append(temp)
        data.append(temp)
        true_label.append([1])
        true_prob.append([1, 0, 0, 0])
    data_1 = np.array(data_1).reshape((l_1, 3))

    for i in range(l_2):
        temp = np.random.multivariate_normal(mu[1], var_2, 1)
        data_2.append(temp)
        data.append(temp)
        true_label.append([2])
        true_prob.append([0, 1, 0, 0])
    data_2 = np.array(data_2).reshape((l_2, 3))

    for i in range(l_3):
        temp = np.random.multivariate_normal(mu[2], var_3, 1)
        data_3.append(temp)
        data.append(temp)
        true_label.append([3])
        true_prob.append([0, 0, 1, 0])
    data_3 = np.array(data_3).reshape((l_3, 3))

    for i in range(l_4):
        temp = np.random.multivariate_normal(mu[3], var_4, 1)
        data_4.append(temp)
        data.append(temp)
        true_label.append([4])
        true_prob.append([0, 0, 0, 1])
    data_4 = np.array(data_4).reshape((l_4, 3))

    data = np.array(data).reshape((N, 3))
    true_prob = np.array(true_prob).reshape((N, 4))
    true_label = np.array(true_label).reshape((N, 1))

    return data, data_1, data_2, data_3, data_4, true_label, true_prob


# In[174]:


l_1, l_2, l_3, l_4 = sample_number(N_1)
data, data_1, data_2, data_3, data_4, true_label, true_prob = data_generator(l_1, l_2, l_3, l_4, mu, var_1, var_2, var_3, var_4)


# In[175]:


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], label='Class 1')
ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], label='Class 2')
ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], label='Class 3')
ax.scatter(data_4[:, 0], data_4[:, 1], data_4[:, 2], label='Class 4')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('Visualization of data as a 2-D scatter plot')
ax.legend()
plt.show()


# ### Points to note:
# 1) The application required that the covariance matrices be symmetric and positive definite for apropriate values of the data points (they were in the gaussian distribution). Hence, to generate such matrices, I have made use of the sklearn.datasets.make_spd_matrix package of sklearn. This package takes the dimensions of the required matrices as arguments and returns a random symmetric, positive-definite matrix.
# 
# 2) Generation of data: Data is generated in two steps:
# 
# Step 1: The number of dataset pairs is calculated of each class using the priors declared (priors are 0.30, 0.25, 0.35, 0.10 for classes 1, 2, 3, and 4 respectively). Uniform distribution is used here to determine the number of pairs for each class.
# 
# Step 2: The data is generated according to Gaussian distribution into four classes based on the size of each class determined from the previos step. The samples are generated according to the conditions mentioned in the question. The means of each of the classes are taken as [2, 2, 0], [-2, 2, 0], [-2, -2, 0], [2, -2, 0] for classes 1, 2, 3, and 4 respectively. This gives a theoretical minimum probability of error between 4-15%. The exact values varies depending on the values generated for the covariences (which are randomly generated as mentioned in point 1).

# ### Part 2:

# In[176]:


l_1, l_2, l_3, l_4 = sample_number(N_2)
data, data_1, data_2, data_3, data_4, true_label, true_prob = data_generator(l_1, l_2, l_3, l_4, mu, var_1, var_2, var_3, var_4)


# In[177]:


l_1, l_2, l_3, l_4


# In[178]:


fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], s=10, label='Class 1')
ax.scatter(data_2[:, 0], data_2[:, 1], data_2[:, 2], s=10, label='Class 2')
ax.scatter(data_3[:, 0], data_3[:, 1], data_3[:, 2], s=10, label='Class 3')
ax.scatter(data_4[:, 0], data_4[:, 1], data_4[:, 2], s=10, label='Class 4')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.title('Visualization of data as a 2-D scatter plot')
ax.legend()
plt.show()


# In[179]:


def normal_prob(x, m, v):
    x_t = x
    m_t = m
    x = np.reshape(x, (3,1))
    m = np.reshape(m, (3,1))
    p = math.exp(-0.5*np.matmul(np.matmul((x_t-m_t), np.linalg.inv(v)), (x-m)))/(2*math.pi*np.linalg.det(v))
    return p


# In[180]:


correct_1 = []
error_1 = []

correct_2 = []
error_2 = []

correct_3 = []
error_3 = []

correct_4 = []
error_4 = []


for i in range(l_1):
    p_1 = normal_prob(data_1[i, :], mu[0], var_1)*prior[0]
    p_2 = normal_prob(data_1[i, :], mu[1], var_2)*prior[1]
    p_3 = normal_prob(data_1[i, :], mu[2], var_3)*prior[2]
    p_4 = normal_prob(data_1[i, :], mu[3], var_4)*prior[3]
    if p_1 >= max(p_2, p_3, p_4):
        correct_1.append(data_1[i, :])
    else:
        error_1.append(data_1[i, :])
correct_1 = np.array(correct_1)
error_1 = np.array(error_1)
    
for i in range(l_2):
    p_1 = normal_prob(data_2[i, :], mu[0], var_1) * prior[0]
    p_2 = normal_prob(data_2[i, :], mu[1], var_2) * prior[1]
    p_3 = normal_prob(data_2[i, :], mu[2], var_3) * prior[2]
    p_4 = normal_prob(data_2[i, :], mu[3], var_4) * prior[3]
    if max(p_1, p_2, p_3, p_4) == p_2:
        correct_2.append(data_2[i, :])
    else:
        error_2.append(data_2[i, :])
correct_2 = np.array(correct_2)
error_2 = np.array(error_2)
        
for i in range(l_3):
    p_1 = normal_prob(data_3[i, :], mu[0], var_1) * prior[0]
    p_2 = normal_prob(data_3[i, :], mu[1], var_2) * prior[1]
    p_3 = normal_prob(data_3[i, :], mu[2], var_3) * prior[2]
    p_4 = normal_prob(data_3[i, :], mu[3], var_4) * prior[3]
    if max(p_1, p_2, p_3, p_4) == p_3:
        correct_3.append(data_3[i, :])
    else:
        error_3.append(data_3[i, :])
correct_3 = np.array(correct_3)
error_3 = np.array(error_3)
        
for i in range(l_4):
    p_1 = normal_prob(data_4[i, :], mu[0], var_1) * prior[0]
    p_2 = normal_prob(data_4[i, :], mu[1], var_2) * prior[1]
    p_3 = normal_prob(data_4[i, :], mu[2], var_3) * prior[2]
    p_4 = normal_prob(data_4[i, :], mu[3], var_4) * prior[3]
    if max(p_1, p_2, p_3, p_4) == p_4:
        correct_4.append(data_4[i, :])
    else:
        error_4.append(data_4[i, :])
correct_4 = np.array(correct_4)
error_4 = np.array(error_4)


# In[181]:


fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(correct_1[:, 0], correct_1[:, 1], correct_1[:, 2], s=10, label='Class 1')
ax.scatter(correct_2[:, 0], correct_2[:, 1], correct_2[:, 2], s=10, label='Class 2')
ax.scatter(correct_3[:, 0], correct_3[:, 1], correct_3[:, 2], s=10, label='Class 3')
ax.scatter(correct_4[:, 0], correct_4[:, 1], correct_4[:, 2], s=10, label='Class 4')
if len(error_1) > 0:
    ax.scatter(error_1[:, 0], error_1[:, 1], error_1[:, 2], s=30, label='Class 1 - Misclassified')
if len(error_2) > 0:
    ax.scatter(error_2[:, 0], error_2[:, 1], error_2[:, 2], s=30, label='Class 2 - Misclassified')
if len(error_3) > 0:
    ax.scatter(error_3[:, 0], error_3[:, 1], error_3[:, 2], s=30, label='Class 3 - Misclassified')
if len(error_4) > 0:
    ax.scatter(error_4[:, 0], error_4[:, 1], error_4[:, 2], s=30, label='Class 4 - Misclassified')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
ax.set_zlabel('Z-axis')
plt.title('Visualization of data according to their true labels and decision labels')
ax.legend()
plt.show()


# In[182]:


print("\nThe total number of samples misclassified by the classifier are {}".format(len(error_1) + len(error_2) + len(error_3) + len(error_4)))

print("\nThe error probability is {}.".format((len(error_1) + len(error_2) + len(error_3) + len(error_4))/N_2))


# In[183]:


print("Wrongly classified points from class 1: {}". format(len(error_1)))
print("Wrongly classified points from class 1: {}". format(len(error_2)))
print("Wrongly classified points from class 1: {}". format(len(error_3)))
print("Wrongly classified points from class 1: {}". format(len(error_4)))


# ### Note: 
# The theory for this part is mentioned at the end of this program.

# ### Part 3:

# In[184]:


N1 = 100
N2 = 1000
N3 = 10000
M = [6, 10, 14, 18, 22, 26, 30, 34, 38]


# In[185]:


l1_1, l1_2, l1_3, l1_4 = sample_number(N1)
data1, data1_1, data1_2, data1_3, data1_4, true_label1, true_prob1 = data_generator(l1_1, l1_2, l1_3, l1_4, mu, var_1, var_2, var_3, var_4)


# In[186]:


l2_1, l2_2, l2_3, l2_4 = sample_number(N2)
data2, data2_1, data2_2, data2_3, data2_4, true_label2, true_prob2 = data_generator(l2_1, l2_2, l2_3, l2_4, mu, var_1, var_2, var_3, var_4)


# In[187]:


l3_1, l3_2, l3_3, l3_4 = sample_number(N3)
data3, data3_1, data3_2, data3_3, data3_4, true_label3, true_prob3 = data_generator(l3_1, l3_2, l3_3, l3_4, mu, var_1, var_2, var_3, var_4)


# In[188]:


def neural(data_train, prob_train, data_test, prob_test, m, N):
    model = Sequential()
    model.add(Dense(m, input_dim=3, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data_train, prob_train, epochs=1000, batch_size=N)
    return model.evaluate(data_test, prob_test)


# In[189]:


def best_model_order(accuracy):
    mean_accuracy = []
    for i in range(len(M)):
        total = 0
        for j in range(10):
            total = total + np.array(accuracy)[i*10 + j, 1]
        mean_accuracy.append(total/10)
    
    param_best = np.array(M)[np.argmax(mean_accuracy)]
    return param_best


# In[190]:


def baseline_model(M=0):
    model = Sequential()
    model.add(Dense(M, input_dim=3, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(4, kernel_initializer='normal', activation='softmax'))
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[191]:


def training(M, N, data, true_label, true_prob):
    parameters = dict(M=M)
    estimator = KerasClassifier(build_fn=baseline_model, epochs=1000, batch_size=N, verbose=0)
    kfold = KFold(n_splits=10, shuffle = True)
    gridsrch = GridSearchCV(estimator = estimator, param_grid = parameters, cv = kfold, n_jobs=-1)
    result = gridsrch.fit(data, true_prob)
    best_score = result.best_score_
    best_param = result.best_params_
    mean = result.cv_results_['mean_test_score']
    std = result.cv_results_['std_test_score']
    param = result.cv_results_['params']  
    
    return best_score, best_param, mean, std, param


# #### 100 dataset:

# In[192]:


best_score_1, best_param_1, mean_1, std_1, param_1 = training(M, N1, data1, true_label1, true_prob1)


# In[193]:


print("The optimal number of perceptrons for minimum error of {} is obtained as {}".format(1-best_score_1, best_param_1))


# In[194]:


for mean, stdev, param in zip(mean_1, std_1, param_1):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[195]:


best_param_1 = best_param_1.get('M')


# In[196]:


accuracy1 = neural(data1, true_prob1, data, true_prob, best_param_1, N1)


# In[197]:


print("The minimum error obtained by applying the above obtained optimal number of perceptrons is {}".format(1-accuracy1[1]))


# #### 1000 dataset:

# In[198]:


best_score_2, best_param_2, mean_2, std_2, param_2 = training(M, N2, data2, true_label2, true_prob2)


# In[199]:


print("The optimal number of perceptrons for minimum error of {} is obtained as {}".format(1-best_score_2, best_param_2))


# In[200]:


for mean, stdev, param in zip(mean_2, std_2, param_2):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[201]:


best_param_2 = best_param_2.get('M')


# In[202]:


accuracy2 = neural(data2, true_prob2, data, true_prob, best_param_2, N2)


# In[203]:


print("The minimum error obtained by applying the above obtained optimal number of perceptrons is {}".format(1-accuracy2[1]))


# #### 10000 dataset:

# In[204]:


best_score_3, best_param_3, mean_3, std_3, param_3 = training(M, N3, data3, true_label3, true_prob3)


# In[205]:


print("The optimal number of perceptrons for minimum error of {} is obtained as {}".format(1-best_score_3, best_param_3))


# In[206]:


for mean, stdev, param in zip(mean_3, std_3, param_3):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[207]:


best_param_3 = best_param_3.get('M')


# In[208]:


accuracy3 = neural(data3, true_prob3, data, true_prob, best_param_3, N3)


# In[209]:


print("The minimum error obtained by applying the above obtained optimal number of perceptrons is {}".format(1-accuracy3[1]))


# ### Points to note (Part 3):
# 1) All the conditions and steps mentioned in the quesion have been adhered to.
# 
# 2) The 3 datasets are generated using the same techniques, distributions, and functions as in part 1 and 2.
# 
# 3) The neural networks have been generated and implemented using the 'keras' package library. All the details mentioned in the question regarding the neural networks (such as the signmoid activation function for the perceptrons, normalized exponential function for the output layer, etc.) have been complied to through this package. The non-specified parameters have been chosen based on common norm for classification problems using neural netwroks. 
# 
# 4) Cross-validation to choose the optimal model order has been implemented manually using a personally created function.
# 
# ### Inferences:
# 1) The test dataset probability errors have been mentioned above at the end of the code.
# 
# 2) These results show that, larger the training dataset size, the better the training of the neural network and thus the neural network classifies the test data more accurately (lesser errors). It is observed that the probability of error of classification of the test data decreases as the training dataset size increases from 100 to 10000.

# In[ ]:




