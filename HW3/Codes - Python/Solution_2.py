#!/usr/bin/env python
# coding: utf-8

# # Question 2:

# In[1102]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
import math
from sklearn.mixture import GaussianMixture


# ### Fisher LDA Classifier

# In[1103]:


s_b = []
s_w = []

mu_1 = [3, 3]
mu_2 = [-3, 3]
variance_1 = [[2, 0.5], [0.5, 1]]
variance_2 = [[2, -1.9], [-1.9, 5]]

prior = [0.30, 0.70]
N = 999
l_1 = 0
l_2 = 0


# In[1104]:


#generating number of sample for the GMMs
for i in range(N):
    if np.random.uniform(0, 1, 1) <= prior[0]:
        l_1 = l_1 + 1

l_2 = N - l_1


# In[1105]:


#generating data according to component
data_1 = []
data_2 = []
data = []
l_i = []
    
for i in range(l_1):
    z = np.random.multivariate_normal(mu_1, variance_1, 1)
    data_1.append(z)
    data.append(z)
    l_i.append(0)
for i in range(l_2):
    z = np.random.multivariate_normal(mu_2, variance_2, 1)
    data_2.append(z)
    data.append(z)
    l_i.append(1)


# In[1106]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(data_1).reshape(l_1,2)[:, 0], np.array(data_1).reshape(l_1,2)[:, 1], s=20, alpha=1, label='Class -')
ax.scatter(np.array(data_2).reshape(l_2,2)[:, 0], np.array(data_2).reshape(l_2,2)[:, 1], s=20, alpha=1, label='Class +')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of datasets and their true classes')
ax.legend()
plt.show()


# In[1107]:


new_mu_1 = [np.array(data_1).reshape(l_1, 2)[:, 0].mean(), np.array(data_1).reshape(l_1, 2)[:, 1].mean()]
new_mu_2 = [np.array(data_2).reshape(l_2, 2)[:, 0].mean(), np.array(data_2).reshape(l_2, 2)[:, 1].mean()]
data_1_t = np.reshape(data_1, (2,l_1))
data_2_t = np.reshape(data_2, (2,l_2))
new_var_1 = np.cov(data_1_t)
new_var_2 = np.cov(data_2_t)


# In[1108]:


s_b = np.matmul(np.subtract(new_mu_1, new_mu_2).reshape(2, 1), (np.subtract(new_mu_1, new_mu_2)).reshape(1, 2))
s_w = np.add(new_var_1, new_var_2)


# In[1109]:


V, D = np.linalg.eig(np.matmul((np.linalg.inv(s_w)), s_b))


# In[1110]:


ind = np.argmax(V)

vec = D[:, ind]
new_ax_1 = np.matmul(vec, np.reshape(data_1, (2, l_1)))
new_ax_2 = np.matmul(vec, np.reshape(data_2, (2, l_2)))


# In[1111]:


tr = 0
err = []
for i in range(l_1):
    count = 0
    tr = new_ax_1[i]
    for j in range(l_1):
        if new_ax_1[j] < tr:
            count = count + 1
    for j in range(l_2):
        if new_ax_2[j] > tr:
            count = count +1
    err.append([tr, count])

for i in range(l_2):
    count = 0
    tr = new_ax_2[i]
    for j in range(l_1):
        if new_ax_1[j] < tr:
            count = count + 1
    for j in range(l_2):
        if new_ax_2[j] > tr:
            count = count +1
    err.append([tr, count])


# In[1112]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(err)[:, 0], np.array(err)[:, 1], s=10)
plt.xlabel('Threshold value')
plt.ylabel('Number of errors')
plt.title('Number of errors at corresponding threshold values')
plt.show()


# In[1113]:


e = np.argmin(np.array(err)[:, 1])
thresh = err[e][0]


# In[1114]:


right_1 = []
error_1 = []
right_2 = []
error_2 = []

for i in range(l_1):
    if new_ax_1[i] > thresh:
        right_1.append(np.array(new_ax_1)[i])
    else:
        error_1.append(np.array(new_ax_1)[i])
        
for i in range(l_2):
    if new_ax_2[i] < thresh:
        right_2.append(np.array(new_ax_2)[i])
    else:
        error_2.append(np.array(new_ax_2)[i])


# In[1115]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(right_1), np.zeros(len(right_1)), s=10, alpha=1, label='Correct classified Label 1')
ax.scatter(np.array(right_2), np.zeros(len(right_2)), s=10, alpha=1, label='Correct classified Label 2')
ax.scatter(np.array(error_1), np.zeros(len(error_1)), s=10, alpha=1, label='Incorrect classified Label 1')
ax.scatter(np.array(error_2), np.zeros(len(error_2)), s=10, alpha=1, label='Incorrect classified Label 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Projection of data points on the projection vector')
ax.legend()
plt.show()


# In[1116]:


print("The total number of errors are: {}".format(len(error_1) + len(error_2)))
print("The error probability is {}.".format((len(error_1) + len(error_2))/N))


# ### Maximum Likelihood Estimation

# In[1126]:


old_parameter = []
data_new = []


# In[1127]:


l_rate = 0.000005

parameter = [vec[0], vec[1], thresh]

l_i = np.reshape(l_i, (N,1))

data = np.reshape(data, (N,2))
data_new = np.hstack((data, np.ones((N,1))))


# In[1140]:


for i in range(150000):
    ex = np.exp(-1*np.matmul(data_new.reshape(N,3), np.array(parameter).reshape(3,1)))
    y_func = np.power(1 + ex, -1)
    deriv = np.matmul(np.reshape(data_new,(3,N)), l_i - y_func.reshape(N,1))
    new_parameter = np.array(parameter).reshape(3,1) - l_rate*deriv
    parameter = new_parameter


# In[1141]:


new_l_i = np.round(np.array(y_func).reshape(N,1))


# In[1142]:


mle_error_1 = []
mle_right_1 = []
mle_error_2 = []
mle_right_2 = []

for i in range(l_1):
    if new_l_i[i] == 1:
        mle_error_1.append(data[i])
    else:
        mle_right_1.append(data[i])
        
for i in range(l_2):
    if new_l_i[l_1+i] == 0:
        mle_error_2.append(data[l_1+i])
    else:
        mle_right_2.append(data[l_1+i])


# In[1143]:


len(mle_error_1), len(mle_right_1), len(mle_error_2), len(mle_right_2)


# In[1144]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(data_1).reshape(l_1,2)[:, 0], np.array(data_1).reshape(l_1,2)[:, 1], s=20, alpha=1, label='Class -')
ax.scatter(np.array(data_2).reshape(l_2,2)[:, 0], np.array(data_2).reshape(l_2,2)[:, 1], s=20, alpha=1, label='Class +')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of datasets and their true classes')
ax.legend()
plt.show()


# In[1145]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(mle_right_1)[:, 0], np.array(mle_right_1)[:, 1], s=20, alpha=1, label='Correct classified Label 1')
ax.scatter(np.array(mle_right_2)[:, 0], np.array(mle_right_2)[:, 1], s=20, alpha=1, label='Correct classified Label 2')
ax.scatter(np.array(mle_error_1)[:, 0], np.array(mle_error_1)[:, 1], s=20, alpha=1, label='Incorrect classified from Label 1 as Label 2')
ax.scatter(np.array(mle_error_2)[:, 0], np.array(mle_error_2)[:, 1], s=20, alpha=1, label='Incorrect classified from Label 2 as Label 1')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of datasets with correct and incorrect classifications')
ax.legend()
plt.show()


# In[1146]:


print("The total number of errors are: {}".format(len(mle_error_1) + len(mle_error_2)))
print("The error probability is {}.".format((len(mle_error_1) + len(mle_error_2))/N))


# ### MAP Classifier

# In[1147]:


def normal_prob(x, m, v):
    x_t = x
    m_t = m
    np.reshape(x, (2,1))
    np.reshape(m, (2,1))
    p = math.exp(-0.5*np.matmul(np.matmul((x_t-m_t), np.linalg.inv(v)), (x-m)))/(2*math.pi*np.linalg.det(v))
    return p


# In[1148]:


x_right = []
x_error = []
y_right = []
y_error = []

for i in range(l_1):
    p_1 = normal_prob(np.array(data)[i, :], mu_1, variance_1)
    p_2 = normal_prob(np.array(data)[i, :], mu_2, variance_2)
    if (p_1*prior[0] > p_2*prior[1]):
        x_right.append(np.array(data)[i, :])
    else:
        x_error.append(np.array(data)[i, :])
        
for i in range(l_2):
    p_1 = normal_prob(np.array(data)[l_1+i, :], mu_1, variance_1)
    p_2 = normal_prob(np.array(data)[l_1+i, :], mu_2, variance_2)
    if (p_1*prior[0] < p_2*prior[1]):
        y_right.append(np.array(data)[l_1+i, :])
    else:
        y_error.append(np.array(data)[l_1+i, :])


# In[1149]:


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(np.array(x_right)[:, 0], np.array(x_right)[:, 1], s=20, alpha=1, label='Correct classified Label 1')
ax.scatter(np.array(y_right)[:, 0], np.array(y_right)[:, 1], s=20, alpha=1, label='Correct classified Label 2')
ax.scatter(np.array(x_error)[:, 0], np.array(x_error)[:, 1], s=20, alpha=1, label='Incorrect classified Label 1')
ax.scatter(np.array(y_error)[:, 0], np.array(y_error)[:, 1], s=20, alpha=1, label='Incorrect classified Label 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph of MAP estimator function')
ax.legend()
plt.show()


# In[1150]:


print("The total number of errors are: {}".format(len(x_error) + len(y_error)))
print("The error probability is {}.".format((len(x_error) + len(y_error))/N))


# ### Results:
# 
# 1) The visual graphs of the results of all the three classifiers have been represnted above. Below the graphs, the respective error counts and the probabilities of error have also been mentioned. 
# 
# 2) For the above dataset, it can be observed here that the least number of errors are generated by the MAP classifier (7), followed by the MLE classifier (24), and finally the Fisher LDA classifier (107). This shows that the MAP classifier gives the best classification for the above dataset closely followed by the MLE classifier.

# In[ ]:




