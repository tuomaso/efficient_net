# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools
import os
import tensorflow as tf
import numpy as np

import preprocessing

class ImageNetInput(object):
  """Base class for ImageNet input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    num_cores: `int` for the number of TPU cores
    image_size: `int` for image size (both width and height).
    transpose_input: 'bool' for whether to use the double transpose trick
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               cache=False):
    self.image_preprocessing_fn = preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.num_cores = 1
    self.transpose_input = transpose_input
    self.image_size = image_size


    ((train_data, train_labels),
    (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()
    
    self.train_data = train_data#/np.float32(255)
    self.train_labels = np.array(train_labels, dtype = np.int32)
    self.eval_data = eval_data#/np.float32(255)
    self.eval_labels = np.array(eval_labels, dtype = np.int32)

  def set_shapes(self, batch_size, images, labels):
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))
    else:
      images.set_shape(images.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
      labels.set_shape(labels.get_shape().merge_with(
          tf.TensorShape([batch_size])))

    return images, labels

  def make_dataset(self):
    if self.is_training:
      train_dataset = tf.data.Dataset.from_tensor_slices((self.train_data,
                                                          self.train_labels))
      train_dataset.shuffle(-1)
      #train_dataset.repeat()
      return train_dataset
    else:
      eval_dataset= tf.data.Dataset.from_tensor_slices((self.eval_data,
                                                        self.eval_labels))
      return eval_dataset

  def resize_helper(self, image, label):
    #print("Resizing: ", image.shape)
    image = tf.expand_dims(image, axis = 2)
    image = tf.image.resize_images(image, (self.image_size, self.image_size))
    return (image, label)
  
  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    dataset = self.make_dataset()
    batch_size = params['batch_size']
    
    '''dataset.apply(
        tf.data.Dataset.batch(batch_size=batch_size,drop_remainder=True))'''
    
    
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.resize_helper, batch_size=batch_size,
            num_parallel_batches=1, drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda images, labels: (tf.transpose(images, [1, 2, 3, 0]), labels),
          num_parallel_calls=self.num_cores)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))
    dataset = dataset.repeat()
    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset



  



