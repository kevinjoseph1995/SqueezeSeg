# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from imdb import kitti,kitti_extended,kitti_extended2,kitti2,kitti_final
from utils.util import *
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeseg/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 50,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('label_format', 'original', """Label Format""")

def train():  
  """Train SqueezeSeg model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'squeezeSeg', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    if FLAGS.net == 'squeezeSeg':
        if FLAGS.label_format=='original':
            mc = kitti_squeezeSeg_config()
        if FLAGS.label_format=='cityscapes':
            mc = kitti_squeezeSeg_config_extended()
        if FLAGS.label_format=='lilanet':
            mc = kitti_squeezeSeg_config_extended2()
        if FLAGS.label_format=='original_new':
            mc = kitti_squeezeSeg_config2()
        if FLAGS.label_format=='original_two_channel':
            mc = kitti_squeezeSeg_config_two_channel()
        if FLAGS.label_format=='final':
            mc = kitti_squeezeSeg_config_final()
        mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
        model = SqueezeSeg(mc)

    if FLAGS.label_format=='original':
      imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
    if FLAGS.label_format=='cityscapes':
      imdb = kitti_extended(FLAGS.image_set, FLAGS.data_path, mc)
    if FLAGS.label_format=='lilanet':
      imdb = kitti_extended2(FLAGS.image_set, FLAGS.data_path, mc)
    if FLAGS.label_format=='original_new':
      imdb = kitti2(FLAGS.image_set, FLAGS.data_path, mc) 
    if FLAGS.label_format=='original_two_channel':
      imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
    if FLAGS.label_format=='final':
      imdb = kitti_final(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    def enqueue(sess, coord):
      with coord.stop_on_exception():
        while not coord.should_stop():
          # read batch input
          lidar_per_batch, lidar_mask_per_batch, label_per_batch,\
              weight_per_batch = imdb.read_batch()

          feed_dict = {
              model.ph_keep_prob: mc.KEEP_PROB,
              model.ph_lidar_input: lidar_per_batch,
              model.ph_lidar_mask: lidar_mask_per_batch,
              model.ph_label: label_per_batch,
              model.ph_loss_weight: weight_per_batch,
          }

          sess.run(model.enqueue_op, feed_dict=feed_dict)
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt==None:
      '''Creating a new Checkpoint'''
      saver = tf.train.Saver(tf.all_variables(),max_to_keep=None)
      summary_op = tf.summary.merge_all()
      init = tf.initialize_all_variables()
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
      sess.run(init)
      global_step=0
    else:
      '''Restoring Checkpoint '''
      var_list=tf.all_variables()
      new_var_list=[variable for variable in var_list if "recurrent_crf" not in variable.name and "conv14_prob" not in variable.name]
      try:
        '''Restoring all variables '''
        check_point_path=ckpt.model_checkpoint_path    
        global_step = int(float(check_point_path.split('/')[-1].split('-')[-1]))    
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=None)
        summary_op = tf.summary.merge_all()
        config=tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        saver.restore(sess, check_point_path)
      except tf.errors.InvalidArgumentError:
        '''Restoring only variables with matching shapes, other variables are randomly initialized'''
        print("###########Number of output channels/labels different from checkpoint. Not restoring the Recurrent CRF Layer and conv14 layer###########")
        check_point_path=ckpt.model_checkpoint_path    
        global_step = int(float(check_point_path.split('/')[-1].split('-')[-1]))    
        saver = tf.train.Saver(new_var_list,max_to_keep=None)
        summary_op = tf.summary.merge_all()
        config=tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        saver.restore(sess, check_point_path)
        '''initializing CRF parameters and conv14 layer'''
        r_crf_var_list=[variable for variable in var_list if "recurrent_crf" in variable.name or "conv14_prob"  in variable.name]
        init_new_vars_op = tf.initialize_variables(r_crf_var_list)
        sess.run(init_new_vars_op)
        '''Setting up global saver'''
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=None)


    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    coord = tf.train.Coordinator()
    enq_threads = []
    for _ in range(mc.NUM_ENQUEUE_THREAD):
      eqth = threading.Thread(target=enqueue, args=[sess, coord])
      eqth.start()
      enq_threads.append(eqth)

    run_options = tf.RunOptions(timeout_in_ms=60000)

    try:
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps-1:
          op_list = [
              model.lidar_input, model.lidar_mask, model.label, model.train_op,
              model.loss, model.pred_cls, summary_op
          ]

          lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
              _, loss_value, pred_cls, summary_str = sess.run(op_list,
                                                              options=run_options)
          label_image = visualize_seg(label_per_batch[:6, :, :], mc)
          pred_image = visualize_seg(pred_cls[:6, :, :], mc)

          # Run evaluation on the batch
          ious, _, _, _ = evaluate_iou(
              label_per_batch, pred_cls*np.squeeze(lidar_mask_per_batch),
              mc.NUM_CLASS)

          feed_dict = {}
          # Assume that class-0 is the background class
          for i in range(1, mc.NUM_CLASS):
            feed_dict[model.iou_summary_placeholders[i]] = ious[i]

          iou_summary_list = sess.run(model.iou_summary_ops[1:], feed_dict)

          # Run visualization
          viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]
          if mc.num_of_input_channels==2:
              viz_summary_list = sess.run(
                  viz_op_list, 
                  feed_dict={
                      model.depth_image_to_show: lidar_per_batch[:6, :, :, [0]],
                      model.label_to_show: label_image,
                      model.pred_image_to_show: pred_image,
                  }
              )
          else:
              viz_summary_list = sess.run(
                  viz_op_list, 
                  feed_dict={
                      model.depth_image_to_show: lidar_per_batch[:6, :, :, [4]],
                      model.label_to_show: label_image,
                      model.pred_image_to_show: pred_image,
                  }
              )


          # Add summaries
          summary_writer.add_summary(summary_str, step)

          for sum_str in iou_summary_list:
            summary_writer.add_summary(sum_str, step)

          for viz_sum in viz_summary_list:
            summary_writer.add_summary(viz_sum, step)

          # force tensorflow to synchronise summaries
          summary_writer.flush()

        else:
          _, loss_value = sess.run(
              [model.train_op, model.loss], options=run_options)

        duration = time.time() - start_time

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                    class_loss)

        if step % 10 == 0:
          num_images_per_step = mc.BATCH_SIZE
          images_per_sec = num_images_per_step / duration
          sec_per_batch = float(duration)
          format_str = ('%s: step %d, loss = %.2f (%.1f images/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,
                               images_per_sec, sec_per_batch))
          sys.stdout.flush()

        # Save the model checkpoint periodically.
        if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps-1:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step+global_step)
    except Exception, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      sess.run(model.q.close(cancel_pending_enqueues=True))
      coord.join(enq_threads)


def main(argv=None):  # pylint: disable=unused-argument
  # if tf.gfile.Exists(FLAGS.train_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.train_dir)
  # tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()