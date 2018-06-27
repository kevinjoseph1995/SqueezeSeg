from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class lilanet(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_lilanet_forward_graph()
      self._add_lilanet_output_graph()
      self._add_lilanet_loss_graph()
      self._add_lilanet_train_graph()
      self._add_lilanet_viz_graph()
      self._add_lilanet_summary_ops()

  def _add_lilanet_forward_graph(self):
    """NN architecture."""

    mc = self.mc 

    lila_1 = self._lila_block('lila_1', self.lidar_input, filters=96)
    lila_2 = self._lila_block('lila_2', lila_1, filters=128)
    lila_3 = self._lila_block('lila_3', lila_2, filters=256)
    lila_4 = self._lila_block('lila_4', lila_3, filters=256)
    lila_5 = self._lila_block('lila_5', lila_4  , filters=128)
    conv6 = self._conv_layer('conv6', lila_5, filters=mc.NUM_CLASS, size=1, stride=1,padding='SAME')
    if mc.num_of_input_channels==2:        
        bilateral_filter_weights = self._bilateral_filter_layer(
            'bilateral_filter', self.lidar_input[:, :, :, :], # inten,depth
            thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
            sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)
    else:
        bilateral_filter_weights = self._bilateral_filter_layer(
            'bilateral_filter', self.lidar_input[:, :, :, :3], # x, y, z
            thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
            sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)
    self.output_prob = self._recurrent_crf_layer(
        'recurrent_crf', conv6, bilateral_filter_weights, 
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
        padding='SAME'
    )
  
  def _lila_block(self,layer_name,inputs,filters):
      """LiLaBlock constructor.
        Args:
        layer_name: layer name
        inputs: input tensor
        filters: number of output filters
        Returns:
        LiLaBlock output with required number of channels
      """
      vert_conv=self._conv_layer_lila(layer_name+'/vert_conv', inputs, filters=filters, size=[7,3], stride=1,padding='SAME')
      symmetric_conv=self._conv_layer_lila(layer_name+'/symmetric_conv', inputs, filters=filters, size=[3,3], stride=1,padding='SAME')
      hori_conv=self._conv_layer_lila(layer_name+'/hori_conv', inputs, filters=filters, size=[3,7], stride=1,padding='SAME') 
      concat_op=tf.concat([vert_conv, symmetric_conv,hori_conv], 3, name=layer_name+'/concat_op')

      return self._conv_layer(layer_name+'/output', concat_op, filters=filters, size=1, stride=1,padding='SAME')

 
