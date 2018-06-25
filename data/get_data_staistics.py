import numpy as np
import os

data_path=os.getcwd()+'/lidar_2d_reduced_classes/'
text_file = open("./ImageSet/allV2.txt", "r")
file_names = text_file.read().split('\n')
x_mean=0.0
x_std=0.0
y_mean=0.0
y_std=0.0
z_mean=0.0
z_std=0.0
i_mean=0.0
i_std=0.0
r_mean=0.0
r_std=0.0
total=len(file_names[:-1])
for name in file_names[:-1]:
    lidar_data=np.load(data_path+name+'.npy')
    x_mean=x_mean+np.mean(np.squeeze(lidar_data[:,:,0]))
    x_std=x_std+np.std(np.squeeze(lidar_data[:,:,0]))
    y_mean=y_mean+np.mean(np.squeeze(lidar_data[:,:,1]))
    y_std=y_std+np.std(np.squeeze(lidar_data[:,:,1]))
    z_mean=z_mean+np.mean(np.squeeze(lidar_data[:,:,2]))
    z_std=z_std+np.std(np.squeeze(lidar_data[:,:,2]))
    i_mean=i_mean+np.mean(np.squeeze(lidar_data[:,:,3]))
    i_std=i_std+np.std(np.squeeze(lidar_data[:,:,3]))
    r_mean=r_mean+np.mean(np.squeeze(lidar_data[:,:,4]))
    r_std=r_std+np.std(np.squeeze(lidar_data[:,:,4]))
x_mean=x_mean/total
x_std=x_std/total
print('x-mean')
print(x_mean)
print('x-std')
print(x_std)
y_mean=y_mean/total
y_std=y_std/total
print('y-mean')
print(y_mean)
print('y-std')
print(y_std)
z_mean=z_mean/total
z_std=z_std/total
print('z-mean')
print(z_mean)
print('z-std')
print(z_std)
i_mean=i_mean/total
i_std=i_std/total
print('i-mean')
print(i_mean)
print('i-std')
print(i_std)
r_mean=r_mean/total
r_std=r_std/total
print('r-mean')
print(r_mean)
print('r-std')
print(r_std)
    