import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

pi= 3.14159265359
def gaussian_pdf(x,mu,sigma): #calculates p(x|mu,sigma) for x1 and x2
    x=x.reshape(-1,1)
    mu=mu.reshape(-1,1)
    p,_=sigma.shape
    
    sigma_inv=np.linalg.inv(sigma)#determinant of sigma
    a=np.sqrt((2*pi)**p*np.linalg.det(sigma)) #first part of gaussian
    b=-0.5*((x-mu).T @ sigma_inv @ (x-mu)) #second part of gaussian

    return float (1./a)*np.exp(b)

def input_data(filename):
    data = pd.read_csv(filename, sep='\s+', header = 0)
    return data

def gauss_dist(x,mu,sigma):
    a=(1/(np.sqrt(2*pi*(sigma**2))))
    b=-(x-mu)**2
    c=2*(sigma**2)
    return a*np.exp(b/c)



data_tr=input_data('synth.tr')
data_tr_c0=data_tr.loc[data_tr.yc==0]# splits data into 2 classes
data_tr_c1=data_tr.loc[data_tr.yc==1]
data_tr_c1=data_tr_c1.reset_index()

n_c0=count_row=data_tr_c0.shape[0]#total number of data points calculation
n_c1=count_row=data_tr_c1.shape[0]
mu_c0_x=(1/n_c0)*data_tr_c0['xs'].sum()#MLE for mu calculation
mu_c0_y=(1/n_c0)*data_tr_c0['ys'].sum()
mu_c1_x=(1/n_c1)*data_tr_c1['xs'].sum()
mu_c1_y=(1/n_c1)*data_tr_c1['ys'].sum()
mu_c0=np.array([mu_c0_x,mu_c0_y])
mu_c1=np.array([mu_c1_x,mu_c1_y])

print("mu_c1=",mu_c1,"\t")
print("mu_c0=",mu_c0,"\t")


sigma_c0_xx=math.sqrt((1/n_c0)*((data_tr_c0['xs']-mu_c0_x)**2).sum())#MLE for sigma calculation
sigma_c0_yy=math.sqrt((1/n_c0)*((data_tr_c0['ys']-mu_c0_y)**2).sum())
sigma_c0_xy=math.sqrt(abs((1/n_c0)*((data_tr_c0['xs']-mu_c0_x)*(data_tr_c0['ys']-mu_c0_y)).sum()))
sigma_c0_yx=math.sqrt(abs((1/n_c0)*((data_tr_c0['ys']-mu_c0_y)*(data_tr_c0['xs']-mu_c0_x)).sum()))

sigma_c1_xx=math.sqrt((1/n_c1)*((data_tr_c1['xs']-mu_c1_x)**2).sum())
sigma_c1_yy=math.sqrt((1/n_c1)*((data_tr_c1['ys']-mu_c1_y)**2).sum())
sigma_c1_xy=math.sqrt(abs((1/n_c1)*((data_tr_c1['xs']-mu_c1_x)*(data_tr_c1['ys']-mu_c1_y)).sum()))
sigma_c1_yx=math.sqrt(abs((1/n_c1)*((data_tr_c1['ys']-mu_c1_y)*(data_tr_c1['xs']-mu_c1_x)).sum()))

sigma_c0=np.array([[ sigma_c0_xx , sigma_c0_xy], [sigma_c0_yx,  sigma_c0_yy]])
sigma_c1=np.array([[ sigma_c1_xx , sigma_c1_xy], [sigma_c1_yx,  sigma_c1_yy]])

print (np.linalg.det(sigma_c0))
print (np.linalg.det(sigma_c1)) 

print("sigma_c0_=",sigma_c0,"\t")
print("sigma_c1_=",sigma_c1,"\t")



X=np.arange(-1.5,1.5,0.01)
Y=np.arange(-1.5,1.5,0.01)
X,Y=np.meshgrid(X, Y)

plt.scatter(data_tr_c0['xs'],data_tr_c0['ys'],c='DarkBlue')# plot data
plt.scatter(data_tr_c1['xs'],data_tr_c1['ys'],c='DarkRed')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1.5,1.5)
plt.xlim(-1.5,1.5)
c_1_leg= mlines.Line2D([], [], color='DarkBlue', marker='*',markersize=15)
c_2_leg= mlines.Line2D([], [], color='DarkRed', marker='*',markersize=15)
plt.legend((c_1_leg,c_2_leg),('class 0', 'class 1'))
plt.title('synth.tr data with decision rules')
plt.show()

