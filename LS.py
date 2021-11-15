# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:12:53 2020

@author: vishn
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import *
import seaborn as sn
import random
'''
#-----------------Task 1--------------------
#Least Square approximation of sin x
x=np.linspace(0,10,num=11)
y=np.sin(x)
A = np.ones((11,1))
for i in range(1,11): 
    A = np.concatenate((A,np.power(x,i).reshape((11,1))),axis=1)
beta_11 = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)

x=np.linspace(0,10,num=101)
y=np.sin(x)
A = np.ones((101,1))
for i in range(1,11): 
    A = np.concatenate((A,np.power(x,i).reshape((101,1))),axis=1)
beta_101 = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)

x=np.linspace(0,10,num=1001)
y=np.sin(x)                            
A = np.ones((1001,1))
for i in range(1,11): 
    A = np.concatenate((A,np.power(x,i).reshape((1001,1))),axis=1)
beta_1001 = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)

x=np.linspace(0,10,num=10001)
y=np.sin(x)                           
A = np.ones((10001,1))
for i in range(1,11): 
    A = np.concatenate((A,np.power(x,i).reshape((10001,1))),axis=1)
beta_10001 = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)


fig, axs = plt.subplots(2, 2)
beta = [[beta_11,beta_101],[beta_1001,beta_10001]]
color = [['blue','orange'],['green','red']]


for i in [0,1]:
    for j in [0,1]:
        y_pred_ls = np.dot(A,beta[i][j])  
        MSE = ((y - y_pred_ls)**2).mean(axis=0)
        print(MSE)
        axs[i, j].plot(x, y_pred_ls, c = color[i][j])

axs[0, 0].set_title('data-size = 11')
axs[0, 1].set_title('data-size = 101')
axs[1, 0].set_title('data-size = 1001')
axs[1, 1].set_title('data-size = 10001')

for ax in fig.get_axes():
    ax.label_outer()
'''
'''
#-----------------Task 2--------------------
#Taylor Series approximation of sin x
x=np.linspace(0,10,num=10001) 
y=np.sin(x)
def taylor_sin(x):
   # return (math.sin(5)+math.cos(5)*(x-5)-(1/math.factorial(2))*math.sin(5)*math.pow(x-5,2)
   #        -(1/math.factorial(3))*math.cos(5)*math.pow(x-5,3)+(1/math.factorial(4))*math.sin(5)*math.pow(x-5,4)
   #        +(1/math.factorial(5))*math.cos(5)*math.pow(x-5,5)-(1/math.factorial(6))*math.sin(6)*math.pow(x-5,6)
   #        -(1/math.factorial(7))*math.cos(5)*math.pow(x-5,7)+(1/math.factorial(8))*math.sin(5)*math.pow(x-5,8)
   #        +(1/math.factorial(9))*math.cos(5)*math.pow(x-5,9)-(1/math.factorial(10))*math.sin(5)*math.pow(x-5,10))
    return (x-(1/math.factorial(3))*pow(x,3)+(1/math.factorial(5))*pow(x,5)-(1/math.factorial(7))*pow(x,7)
            +(1/math.factorial(9))*pow(x,9))
y_pred_tay = np.zeros((10001,))
for i in range(10001):
    y_pred_tay[i] = taylor_sin(x[i])
   
MSE = ((y - y_pred_tay)**2).mean(axis=0)    
print(MSE)
plt.plot(x,y,color = 'red', label = 'sin(x)')
plt.plot(x,y_pred_tay, color = 'blue', label='taylor approximation')  
plt.xlabel('x')
plt.legend(loc='upper right') 
'''
'''
#-----------------Task 3--------------------
#best appeoximation method for sinx
x=np.linspace(0,10,num=10001) 
y=np.sin(x)
z = symbols('z')
basis = np.array([1,z,pow(z,2),pow(z,3),pow(z,4),pow(z,5)])
def innerproduct(f,g):
    return integrate(f*g,(z,0,10))

def gramschmidt(basis):
    new_basis = []
    new_basis.append(basis[0])
    
    for i in range(1, basis.shape[0]):
        summ = 0
        for j in range(0, i):
            summ += innerproduct(basis[i],new_basis[j])/innerproduct(new_basis[j],new_basis[j])*new_basis[j]
        new_basis.append(basis[i]-summ)
    return new_basis     

ortho_basis = gramschmidt(basis)

def best_approx(f,ortho_basis):
    b = 0
    for i in range(0,len(ortho_basis)):
        b += innerproduct(f, ortho_basis[i])/innerproduct(ortho_basis[i], ortho_basis[i])*ortho_basis[i]
    return b 

y_best = best_approx(sin(z), ortho_basis)
y_pred_best = np.zeros((10001,))
for i in range(10001):
    y_pred_best[i] = y_best.subs(z,x[i]).evalf()
MSE = ((y - y_pred_best)**2).mean(axis=0)    
print(MSE)
plt.plot(x,y,color = 'red',label = 'sin(x)')
plt.plot(x,y_pred_best, color = 'blue', label='best approximation')  
plt.xlabel('x')  
plt.legend(loc='upper right') 
'''
'''
#-----------------Task 4--------------------
# Vandermonde function
def VanderMonde(x,P):
    n = P
    m = x.shape[0]
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(1,n+1):
            A[i,j-1] = x[i]**(j-1)
    return A

x = np.array([1,2,3,4,5,6])
y = np.array([2,3,5,7,11,13])
P=3
A = VanderMonde(x,P)   
 
#-----------------Task 5-------------------- 
beta = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y)  
y_pred_ls = np.dot(A,beta) 
MSE_ls = ((y - y_pred_ls)**2).mean(axis=0)
print(MSE_ls)
x1 = np.linspace(0,7,num=101)
A1 = VanderMonde(x1,P) 
y_pred_ls1 = np.dot(A1,beta)
plt.scatter(x,y,color='k',label='given points')  
plt.plot(x1,y_pred_ls1,color='r',label = 'least-square solution')
plt.xlabel('x')
plt.ylabel('y(degree-2 polynomial approximation)')


#-----------------Task 6--------------------
N = x.shape[0]-1
t = 0
beta_t = np.zeros((P,))
y_pred_gr = np.dot(A,beta_t) 
MSE_gr = ((y - y_pred_gr)**2).mean(axis=0)
gamma = 0.0002
while MSE_gr > MSE_ls+0.2*MSE_ls:
    t = t+1
    beta_t1  = beta_t + gamma*(y[t%N+1]-np.dot(A[t%N+1],beta_t))*A[t%N+1]
    beta_t = beta_t1
    y_pred_gr = np.dot(A,beta_t) 
    MSE_gr = ((y - y_pred_gr)**2).mean(axis=0)
    
print(t)
y_pred_gr = np.dot(A,beta_t) 
MSE_gr = ((y - y_pred_gr)**2).mean(axis=0)
print(MSE_gr)
#x = np.linspace(0,7,num=101)
#A = VanderMonde(x,P) 
y_pred_gr = np.dot(A1,beta_t)  
plt.plot(x1,y_pred_gr,color='b',label = 'gradient solution')    
plt.legend(loc='upper right')
'''

#-----------------Task 7--------------------
import pandas as pd


# Read MNIST c s v f i l e i n t o da ta frame
df = pd.read_csv('mnist_train.csv')
# Merge p i x e l s i n t o f e a t u r e column and keep o nly f e a t u r e and l a b e l
df['feature'] = df.apply(lambda row : row.values[1:] ,axis=1)
df = df[['feature', 'label']]

'''
# Pl o t MNIST
plt.figure(figsize =(15, 2.5))
for i, row in df.iloc[:30].iterrows():
    x,y=row['feature'], row['label']
    plt.subplot(2,15,i+1)
    plt.imshow(x.reshape(28,28),cmap='gray')
    plt.axis('off')
    plt.title(y)
''' 
'''  
zero_data = np.array(df['feature'][df.index[df['label']==0]].to_list())
zero_data=np.insert(zero_data,0,-1,axis=1)
s1 = np.random.choice(range(zero_data.shape[0]), math.ceil(zero_data.shape[0]/2), replace=False)
s2 = list(set(range(zero_data.shape[0])) - set(s1))

zero_train = zero_data[s1,:]
zero_train_label = -1*np.ones((zero_train.shape[0],))
zero_test = zero_data[s2,:]
zero_test_label = -1*np.ones((zero_test.shape[0],))

one_data = np.array(df['feature'][df.index[df['label']==1]].to_list())
one_data=np.insert(one_data,0,-1,axis=1)
s1 = np.random.choice(range(one_data.shape[0]), math.ceil(one_data.shape[0]/2), replace=False)
s2 = list(set(range(one_data.shape[0])) - set(s1))

one_train = one_data[s1,:]
one_train_label = np.ones((one_train.shape[0],))
one_test = one_data[s2,:]
one_test_label = np.ones((one_test.shape[0],))

A = np.concatenate((zero_train,one_train),axis=0)
y_train = np.concatenate((zero_train_label,one_train_label),axis=0)
s1 = np.random.choice(range(A.shape[0]), A.shape[0], replace=False)
A = A[s1,:]
y_train = y_train[s1]

#LS Method
#beta = np.dot(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),y_train)
beta = np.linalg.lstsq(A,y_train,rcond=None)[0]
y_est = np.dot(A,beta) 
MSE_ls = ((y_train - y_est)**2).mean(axis=0)
print(MSE_ls)

#Gradient method
N = A.shape[0]-1
t = 0
beta_t = np.zeros((A.shape[1],))
y_pred_gr = np.dot(A,beta_t) 
MSE_gr = ((y_train - y_pred_gr)**2).mean(axis=0)
gamma = 2e-7
while MSE_gr > 2*MSE_ls:
    beta_t1  = beta_t + gamma*(y_train[t%N+1]-np.dot(A[t%N+1],beta_t))*A[t%N+1]
    #beta_t1  = beta_t + gamma*np.dot(A.T,(y_train-np.dot(A,beta_t)))
    beta_t = beta_t1
    y_pred_gr = np.dot(A,beta_t) 
    MSE_gr = ((y_train - y_pred_gr)**2).mean(axis=0)
    t = t+1
    #print(MSE_gr)
    
y_pred_gr = np.dot(A,beta_t) 
MSE_gr = ((y_train - y_pred_gr)**2).mean(axis=0)
print(MSE_gr)
beta = beta_t


ztraine = np.dot(zero_train,beta)
ztrain_w=np.count_nonzero(ztraine>0)
ztrain_c=np.count_nonzero(ztraine<0)
zteste = np.dot(zero_train,beta)
ztest_w=np.count_nonzero(zteste>0)
ztest_c=np.count_nonzero(zteste<0)

otraine = np.dot(one_train,beta)
otrain_c=np.count_nonzero(otraine>0)
otrain_w=np.count_nonzero(otraine<0)
oteste = np.dot(one_test,beta)
otest_c=np.count_nonzero(oteste>0)
otest_w=np.count_nonzero(oteste<0)

#Classification error rate
print('Classification error rate for 0-1 class(train data):',(otrain_w+ztrain_w)/(ztrain_w+ztrain_c+otrain_c+otrain_w))
print('Classification error rate for 0-1 class(test data):',(otest_w+ztest_w)/(ztrain_w+ztrain_c+otrain_c+otrain_w))

#Confusion-Matrix
C_train = np.array([[ztrain_c,ztrain_w],[otrain_w,otrain_c]])
C_test = np.array([[ztest_c,ztest_w],[otest_w,otest_c]])
df_cm = pd.DataFrame(C_train, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (8,8))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.ylabel('actual-class')
plt.xlabel('predicted-class')

df_cm = pd.DataFrame(C_test, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (8,8))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.ylabel('actual-class')
plt.xlabel('predicted-class')
#print(MSE)

plt.figure(figsize = (8,8))
x = [-1.25, 0.75]
y= [-1,1]
plt.hist(x, bins = [-1.25,-1,0.75,1], weights = [ztest_c,ztest_w],color='blue',label='class-0=-1')
plt.hist(y, bins = [-1,-0.75,1,1.25], weights = [otest_w,otest_c],color='red',label='class-1=1')
plt.legend(loc='upper right')
plt.xlabel('class')
plt.ylabel('number of points identified in each class')
plt.show()
'''
'''
#-----------------Task 8--------------------
data = []
data_train = []
data_train_label = []
data_test = []
data_test_label =[]

for i in range(10):
    data.append(np.array(df['feature'][df.index[df['label']==i]].to_list()))
    s1 = np.random.choice(range(data[i].shape[0]), math.ceil(data[i].shape[0]/2), replace=False)
    s2 = list(set(range(data[i].shape[0])) - set(s1))
    data_train.append(data[i][s1,:])
    data_test.append(data[i][s2,:])

mask = np.zeros((10,10))
for i in range(10):
    mask[i,i]=1
M = np.zeros((10,10))    
for i in range(10):    
    for j in range(i+1,10):
        A = np.concatenate((data_train[i],data_train[j]),axis=0)
        y_train = np.concatenate((-1*np.ones((data_train[i].shape[0],)),np.ones((data_train[j].shape[0],))),axis=0)
        beta = np.linalg.lstsq(A,y_train,rcond=None)[0]
        itraine = np.dot(data_train[i],beta)
        itrain_w=np.count_nonzero(itraine>0)
        itrain_c=np.count_nonzero(itraine<0)
        iteste = np.dot(data_test[i],beta)
        itest_w=np.count_nonzero(iteste>0)
        itest_c=np.count_nonzero(iteste<0)
        
        jtraine = np.dot(data_train[j],beta)
        jtrain_c=np.count_nonzero(jtraine>0)
        jtrain_w=np.count_nonzero(jtraine<0)
        jteste = np.dot(data_test[j],beta)
        jtest_c=np.count_nonzero(jteste>0)
        jtest_w=np.count_nonzero(jteste<0)
        M[i,j] = round((itrain_w+jtrain_w)/(itrain_w+itrain_c+jtrain_c+jtrain_w),4)
        M[j,i] = round((itest_w+jtest_w)/(itest_w+itest_c+jtest_c+jtest_w),4)

M = np.ma.masked_array(M,mask)
df_cm = pd.DataFrame(M, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Greens', fmt='g')
#plt.ylabel('actual-class')
#  plt.xlabel('predicted-class')
'''

#-----------------Task 9--------------------
data = np.array(df['feature'].to_list())
data_label = np.array(df['label'].to_list())
#print(zero_data.shape)
data=np.insert(data,0,-1,axis=1)
#print(zero_data.shape)
s1 = np.random.choice(range(data.shape[0]), math.ceil(data.shape[0]/2), replace=False)
s2 = list(set(range(data.shape[0])) - set(s1))

train_data = data[s1,:]
train_label = data_label[s1]
test_data = data[s2,:]
test_label = data_label[s2]
Y = np.zeros((train_label.shape[0],10))
for i in range(train_label.shape[0]):
    Y[i,train_label[i]]=1
A = train_data    
beta = np.linalg.lstsq(A,Y,rcond=None)[0]    

dtraine = np.argmax(np.dot(train_data,beta),axis=1)
dteste = np.argmax(np.dot(test_data,beta),axis=1)
#np.count_nonzero((dtraine==train_label)==True)
#Confusion Matrix
C_train = np.zeros([10,10])
C_test = np.zeros([10,10])

for i in range(10):
    train_label_val = np.array(np.where(train_label==i))
    for j in range(10):
        C_train[i,j] = np.count_nonzero(dtraine[train_label_val]==j)
for i in range(10):
    test_label_val = np.array(np.where(test_label==i))
    for j in range(10):
        C_test[i,j] = np.count_nonzero(dteste[test_label_val]==j)        

        
df_cm = pd.DataFrame(C_train, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.ylabel('actual-class')
plt.xlabel('predicted-class') 

df_cm = pd.DataFrame(C_test, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.ylabel('actual-class')
plt.xlabel('predicted-class')        
