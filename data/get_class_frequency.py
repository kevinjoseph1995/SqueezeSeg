import numpy as np
import os

data_path=os.getcwd()+'/lidar_2d_final/'
text_file = open("./ImageSet/allV2.txt", "r")
file_names = text_file.read().split('\n')

total=len(file_names[:-1])
#CLASSES= ['unknown', 'road', 'sidewalk', 'building','wall','fence','pole','traffic-light','traffic-sign','vegetation','terrain','sky','person',\
#                               'rider','car','truck','bus','train','motorcycle','bicycle']
CLASSES=['unknown', 'road', 'sidewalk', 'construction','vegetation','terrain','person','small-vehicle','large-vehicle','two-wheeler'] 
total_cont=np.zeros([len(CLASSES)],dtype='int')                        
for name in file_names[:-1]:
    lidar_data=np.load(data_path+name+'.npy')    
    total_cont+=np.bincount(lidar_data[:,:,5].astype('int').flatten(),minlength=len(CLASSES))
normalized_count=total_cont/np.max(total_cont)
print('Actual frequency')
for pair in zip(CLASSES,total_cont):
    print(pair)
print('Normalized frequency')
for pair in zip(CLASSES,normalized_count):
    print(pair)
print(normalized_count)
learning_rate_setting=1.0/total_cont
print(learning_rate_setting/np.max(learning_rate_setting))
    