Datasets explained.
1) lidar_2d- Original SqueezeSeg data. 3 classes(car,person and cyclist)
2) lidar_2d_new- Generated Ground-truth with 19 classes(according to Cityscape)
3) lidar_2d_reduced_classes-Generated Ground-truth with 12 classes, some classes have been fused and 'sky' class has been discarded due to rare occurence. Follows labelling scheme according to LiLaNet.
4) lidar_2d_new_SqSeg_classes- Generated Ground-truth with the 3 classes from the orignal SqueezeSeg paper. Created to test the credibility of the generated ground-truth. 
