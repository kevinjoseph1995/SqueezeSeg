# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71],
                                    [ 0.58,  0.72,  0.88]])

  mc.BATCH_SIZE         = 32
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])
  mc.num_of_input_channels=5
  mc.use_focal_loss=False

  return mc

def kitti_squeezeSeg_config_extended():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'road', 'sidewalk', 'building','wall','fence','pole','traffic-light','traffic-sign','vegetation','terrain','sky','person',\
                            'rider','car','truck','bus','train','motorcycle','bicycle']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([  7.64682756e-05,   2.50118603e-04,   7.08845536e-04,   9.48929953e-04,   2.70868307e-02,   4.51554217e-03,   9.98267695e-03,   3.94562539e-01,\
                                      3.77046043e-02,   4.94698981e-04,   1.18539107e-03,   1.00000000e+00,   1.29004315e-02,   3.90532093e-01,   1.09553689e-03,   3.11450345e-02,\
                                      2.08502065e-01,   6.15127846e-02,   1.66865355e-02,   3.90703350e-02])
  
  mc.CLS_COLOR_MAP      = np.array([[ 0,  0,  0],
                                    [ 128,64,128],
                                    [ 244,35,232],
                                    [ 70, 70, 70],
                                    [ 102,102,156],
                                    [ 190,153,153],
                                    [ 153,153,153],
                                    [ 250,170,30],
                                    [ 220,220,0],
                                    [ 107,142,35],
                                    [ 152,251,152],
                                    [ 70,130,180],
                                    [ 220,20,60],
                                    [ 255,0,0],
                                    [ 0,0,142],
                                    [ 0,0,70],
                                    [ 0,60,100],
                                    [ 0,80,100],
                                    [ 0,0,230],                                
                                    [ 0,0,230]])

  mc.BATCH_SIZE         = 16
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  #unsure whether to be changed
  mc.BILATERAL_THETA_A  = 0.01/np.array([  7.64682756e-05,   2.50118603e-04,   7.08845536e-04,   9.48929953e-04,   2.70868307e-02,   4.51554217e-03,   9.98267695e-03,   3.94562539e-01,\
                                      3.77046043e-02,   4.94698981e-04,   1.18539107e-03,   1.00000000e+00,   1.29004315e-02,   3.90532093e-01,   1.09553689e-03,   3.11450345e-02,\
                                      2.08502065e-01,   6.15127846e-02,   1.66865355e-02,   3.90703350e-02])
  mc.BILATERAL_THETA_R  = 100*np.array([  7.64682756e-05,   2.50118603e-04,   7.08845536e-04,   9.48929953e-04,   2.70868307e-02,   4.51554217e-03,   9.98267695e-03,   3.94562539e-01,\
                                      3.77046043e-02,   4.94698981e-04,   1.18539107e-03,   1.00000000e+00,   1.29004315e-02,   3.90532093e-01,   1.09553689e-03,   3.11450345e-02,\
                                      2.08502065e-01,   6.15127846e-02,   1.66865355e-02,   3.90703350e-02])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = 0.01/np.array([  7.64682756e-05,   2.50118603e-04,   7.08845536e-04,   9.48929953e-04,   2.70868307e-02,   4.51554217e-03,   9.98267695e-03,   3.94562539e-01,\
                                      3.77046043e-02,   4.94698981e-04,   1.18539107e-03,   1.00000000e+00,   1.29004315e-02,   3.90532093e-01,   1.09553689e-03,   3.11450345e-02,\
                                      2.08502065e-01,   6.15127846e-02,   1.66865355e-02,   3.90703350e-02])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 1.17
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance(Recomputed for the extended dataset. )
  mc.INPUT_MEAN         = np.array([[[7.11174452147, 0.0943776729418, -0.500243253951, 7.70754909688, 0.110507476635]]])
  mc.INPUT_STD          = np.array([[[10.9252722964, 4.8862697957, 0.775360202555, 11.6589714541, 0.166859600962]]])
  mc.num_of_input_channels=5
  mc.use_focal_loss=False

  return mc

def kitti_squeezeSeg_config_extended2():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'road', 'sidewalk', 'construction','pole','traffic-sign','vegetation','terrain','person','rider','small-vehicle','large-vehicle','two-wheeler']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([  1.92530238e-04,   6.40455949e-04,   1.81507627e-03,   2.34215328e-03,
                                      2.55617326e-02,   9.65467499e-02,   1.26673067e-03,   3.03532306e-03,
                                      3.30329612e-02,   1.00000000e+00,   2.80524164e-03,   4.81672894e-02,
                                      2.99404409e-02])
  
  mc.CLS_COLOR_MAP      = np.array([[ 0,  0,  0],
                                    [ 128,64,128],
                                    [ 244,35,232],
                                    [ 70, 70, 70],
                                    [ 153,153,153],
                                    [ 220,220,0],
                                    [ 107,142,35],
                                    [ 152,251,152],
                                    [ 220,20,60],                                    
                                    [ 255,0,0],
                                    [ 0,0,142],
                                    [ 0,0,70],
                                    [ 0,0,230]])

  mc.BATCH_SIZE         = 32
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  #unsure whether to be changed
  mc.BILATERAL_THETA_A  =0.01/ np.array([  1.92530238e-04,   6.40455949e-04,   1.81507627e-03,   2.34215328e-03,
                                      2.55617326e-02,   9.65467499e-02,   1.26673067e-03,   3.03532306e-03,
                                      3.30329612e-02,   1.00000000e+00,   2.80524164e-03,   4.81672894e-02,
                                      2.99404409e-02])
  mc.BILATERAL_THETA_R  = 100*np.array([  1.92530238e-04,   6.40455949e-04,   1.81507627e-03,   2.34215328e-03,
                                      2.55617326e-02,   9.65467499e-02,   1.26673067e-03,   3.03532306e-03,
                                      3.30329612e-02,   1.00000000e+00,   2.80524164e-03,   4.81672894e-02,
                                      2.99404409e-02])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = 0.01/ np.array([  1.92530238e-04,   6.40455949e-04,   1.81507627e-03,   2.34215328e-03,
                                      2.55617326e-02,   9.65467499e-02,   1.26673067e-03,   3.03532306e-03,
                                      3.30329612e-02,   1.00000000e+00,   2.80524164e-03,   4.81672894e-02,
                                      2.99404409e-02])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance(Recomputed for the extended dataset. )
  mc.INPUT_MEAN         = np.array([[[7.11174452147, 0.0943776729418, -0.500243253951, 0.110507476635,7.70754909688]]])
  mc.INPUT_STD          = np.array([[[10.9252722964, 4.8862697957, 0.775360202555, 0.166859600962,11.6589714541]]])
  mc.num_of_input_channels=5
  mc.use_focal_loss=True

  return mc

def kitti_squeezeSeg_config2():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71],
                                    [ 0.58,  0.72,  0.88]])

  mc.BATCH_SIZE         = 32
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance(Recomputed for the extended dataset. )
  mc.INPUT_MEAN         = np.array([[[7.11174452147, 0.0943776729418, -0.500243253951, 0.110507476635,7.70754909688]]])
  mc.INPUT_STD          = np.array([[[10.9252722964, 4.8862697957, 0.775360202555, 0.166859600962,11.6589714541]]])
  mc.num_of_input_channels=5
  mc.use_focal_loss=False

  return mc

def kitti_squeezeSeg_config_two_channel():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([1/15.0, 1.0,  10.0, 10.0])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71],
                                    [ 0.58,  0.72,  0.88]])

  mc.BATCH_SIZE         = 32
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.01
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])
  mc.num_of_input_channels=2
  mc.use_focal_loss=False

  return mc


