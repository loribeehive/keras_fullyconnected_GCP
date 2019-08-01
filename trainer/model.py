# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implements the Keras Sequential model."""

from builtins import range

import keras
import pathlib
from keras import backend as K
from keras import layers
from keras import models
from keras.backend import relu
import argparse
import pandas as pd
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


import numpy as np
bins = np.array([50,100,150,200,250,500,1100])
CSV_COLUMNS = ('time','IOPS','Throughput','MBps')

one_hour = 12

def model_fn(input_dim,
             labels_dim,
             hidden_units,
             learning_rate=0.1):
  """Create a Keras Sequential model with layers.

  Args:
    input_dim: (int) Input dimensions for input layer.
    labels_dim: (int) Label dimensions for input layer.
    hidden_units: [int] the layer sizes of the DNN (input layer first)
    learning_rate: (float) the learning rate for the optimizer.

  Returns:
    A Keras model.
  """

  # "set_learning_phase" to False to avoid:
  # AbortionError(code=StatusCode.INVALID_ARGUMENT during online prediction.
  K.set_learning_phase(False)
  model = models.Sequential()
  hidden_units = [int(units) for units in hidden_units.split(',')]

  for units in hidden_units:
      model.add(layers.Dense(units, activation=relu, input_shape=[input_dim],
                            kernel_initializer='glorot_uniform',
                            ))
      model.add(layers.Dropout(0.5))
      model.add(layers.BatchNormalization(epsilon=1e-03, momentum=0.9, weights=None))
      input_dim = units
      #                 activity_regularizer=tf.keras.regularizers.l1(0.01)


  # Add a dense final layer with sigmoid function.
  model.add(layers.Dense(labels_dim, activation='softmax'))
  compile_model(model, learning_rate)
  return model

# def _construct_hidden_units(hidden_units):
#   """ Create the number of hidden units in each layer
#   if the args.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
#   to define the number of units in each layer. Otherwise, arg.hidden_units
#   will be used as-is.
#   Returns:
#       list of int
#   """
#   hidden_units = [int(units) for units in hidden_units.split(',')]
#
#
#
#
#   return hidden_units


def compile_model(model, learning_rate):
  model.compile(
      loss='categorical_crossentropy',
      optimizer=keras.optimizers.Adam(lr=learning_rate),
      metrics=['accuracy'])
  return model


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)
  signature = predict_signature_def(
      inputs={'MBps': model.inputs[0]}, outputs={'Category': model.outputs[0]}
  )
  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
    )
    builder.save()


def generator_input(filenames, training_history, batch_size):
  """Produce features and labels needed by keras fit_generator."""
  # if tf.gfile.IsDirectory(filenames):
  #     files = tf.gfile.ListDirectory(filenames)
  # else:
  #     files = filenames

  while True:
      # data = np.load(filenames, allow_pickle=True).item()
      # keys = list(data.keys())
      files = tf.gfile.Glob(filenames)
      # path = pathlib.Path(filenames)
      # files = path.iterdir()
      for file in files:
          file = str(file)
          input_reader = pd.read_csv(
            tf.gfile.Open(file),
            names=CSV_COLUMNS,
            skiprows=1,
            na_values=' ?')

          input_data = input_reader.dropna()
          (input,label)=process_data(input_data,training_history=training_history)
          if input.shape[0]<2 or label.shape[0]<2:
              continue
          idx_len = input.shape[0]
          for index in range(0, idx_len, batch_size):
            yield (input[index:min(idx_len, index + batch_size)],
                   label[index:min(idx_len, index + batch_size)])



def process_data(data,training_history):


    data_read = data[['time', 'MBps']]
    # insert label
    data_read['Bucket_Index'] = np.digitize(data_read['MBps'], bins, right=False)
    # make time series as TimeIndex
    idx = pd.to_datetime(data_read['time'])
    data_read = data_read.drop(['time'], axis=1)
    data_read = data_read.set_index(idx)

    ####take the mean of every five mins
    data_sample = data_read['MBps'].resample('5Min').mean()
    data_sample = data_sample.fillna(0)
    # resample label to max(1H)
    label_data = data_read['Bucket_Index'].resample('1H').max()
    label_data  = label_data .fillna(0)

    ############split test data by day###########
    # date_range = idx.iloc[-1]-idx.iloc[0]
    # if date_range>pd.Timedelta('8 days'):
    #     testEnd  = date_range*0.2
    #     labelEndIndex = label_data.index.get_loc(idx.iloc[0]+testEnd, method='nearest')
    #     dataEndIndex = data_sample.index.get_loc(idx.iloc[0] + testEnd, method='nearest')
    # else:
    #     labelEndIndex = label_data.index.get_loc(idx.iloc[0] + pd.Timedelta('1 days'), method='nearest')
    #     dataEndIndex = data_sample.index.get_loc(idx.iloc[0] + pd.Timedelta('1 days'), method='nearest')
    # ############test###########
    # disk_testX,disk_testY = reshape_input(data_sample.iloc[0:dataEndIndex],label_data.iloc[0:labelEndIndex],training_history)
    if data_sample.empty or label_data.empty:
        return (np.array([]),np.array([]))

    disk_trainX,disk_trainY = reshape_input(data_sample, label_data,
                                   training_history)


    return (disk_trainX,disk_trainY)



def reshape_input(data, label,training_history):
    data_SIZE = len(label.values)-1
    subLabel = np.zeros((data_SIZE, len(bins)))
    label=label.iloc[:-1]
    subLabel[np.arange(data_SIZE), np.array(label.values).astype(int)] = 1
    subLabel = subLabel[training_history:, :]

    LENN = subLabel.shape[0]
    subData = np.array(data.values)
    subData = subData[0:one_hour * (LENN + (training_history - 1))]
    s_Data=np.array([])
    for j in range(LENN):

        s_data = subData[j * one_hour:(j * one_hour + one_hour * training_history)]
        if j == 0:
            s_Data = s_data
            #####if only one row of data in total
            s_Data = s_Data.reshape(-1, one_hour * training_history)

        else:
            s_Data = np.vstack((s_Data, s_data))


    return (s_Data, subLabel)








###############################Google Code###############################################

# CSV columns in the input file.
# CSV_COLUMNS = ('age', 'workclass', 'fnlwgt', 'education', 'education_num',
#                'marital_status', 'occupation', 'relationship', 'race', 'gender',
#                'capital_gain', 'capital_loss', 'hours_per_week',
#                'native_country', 'income_bracket')
#
# CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
#                        [0], [0], [0], [''], ['']]
#
# # Categorical columns with vocab size
# # native_country and fnlwgt are ignored
# CATEGORICAL_COLS = (('education', 16), ('marital_status', 7),
#                     ('relationship', 6), ('workclass', 9), ('occupation', 15),
#                     ('gender', [' Male', ' Female']), ('race', 5))
#
# CONTINUOUS_COLS = ('age', 'education_num', 'capital_gain', 'capital_loss',
#                    'hours_per_week')
#
# LABELS = [' <=50K', ' >50K']
# LABEL_COLUMN = 'income_bracket'
#
# UNUSED_COLUMNS = set(CSV_COLUMNS) - set(
#     list(zip(*CATEGORICAL_COLS))[0] + CONTINUOUS_COLS + (LABEL_COLUMN,))

# def to_numeric_features(features, feature_cols=None):
#   """Converts the pandas input features to numeric values.
#
#   Args:
#     features: Input features in the data age (continuous) workclass
#       (categorical) fnlwgt (continuous) education (categorical) education_num
#       (continuous) marital_status (categorical) occupation (categorical)
#       relationship (categorical) race (categorical) gender (categorical)
#       capital_gain (continuous) capital_loss (continuous) hours_per_week
#       (continuous) native_country (categorical)
#     feature_cols: Column list of converted features to be returned. Optional,
#       may be used to ensure schema consistency over multiple executions.
#
#   Returns:
#     A pandas dataframe.
#   """
#
#   for col in CATEGORICAL_COLS:
#     features = pd.concat(
#         [features, pd.get_dummies(features[col[0]], drop_first=True)], axis=1)
#     features.drop(col[0], axis=1, inplace=True)
#
#   # Remove the unused columns from the dataframe.
#   for col in UNUSED_COLUMNS:
#     features.pop(col)
#
#   # Re-index dataframe (if categories list changed from the previous dataset)
#   if feature_cols is not None:
#     features = features.T.reindex(feature_cols).T.fillna(0)
#   return features
#
#
# def generator_census_input(filenames, chunk_size, batch_size=64):
#   """Produce features and labels needed by keras fit_generator."""
#
#   feature_cols = None
#   while True:
#     input_reader = pd.read_csv(
#         tf.gfile.Open(filenames[0]),
#         names=CSV_COLUMNS,
#         chunksize=chunk_size,
#         na_values=' ?')
#
#     for input_data in input_reader:
#       input_data = input_data.dropna()
#       label = pd.get_dummies(input_data.pop(LABEL_COLUMN))
#
#       input_data = to_numeric_features(input_data, feature_cols)
#
#       # Retains schema for next chunk processing.
#       if feature_cols is None:
#         feature_cols = input_data.columns
#
#       idx_len = input_data.shape[0]
#       for index in range(0, idx_len, batch_size):
#         yield (input_data.iloc[index:min(idx_len, index + batch_size)],
#                label.iloc[index:min(idx_len, index + batch_size)])
