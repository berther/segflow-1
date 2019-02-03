"""Weighted mean iou metric"""
import numpy as np
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf


def mean_iou_func(y_true, y_pred, batch_size, dims, n_classes=20,
                  weight_vector=None, iou_test=False, one_hot=True):
    if one_hot:
        # convert onehot back to label ints
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

    # reshape
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    # print("ytrue", y_true.name, y_true.shape)
    # print("ypred", y_pred.name, y_pred.shape)

    # weights have to be computed here differently for each batch
    y_true_int = tf.cast(y_true, tf.int32)
    if weight_vector is not None:
        weight_vector = tf.constant(np.array(weight_vector).astype(np.int32))
        # print("iou weight vector", weight_vector)

        # # the numpy way
        # weights_matrix = weight_vector[y_true_int]

        # tensorflow way
        weight_matrix = tf.gather(weight_vector, y_true_int)
        # print("tf.gather result", weight_matrix )
        weight_matrix = tf.reshape(weight_matrix, (batch_size, dims[0] * dims[1]))
        weight_matrix = tf.layers.Flatten()(weight_matrix)

        # print("iou weight matrix shape", weight_matrix.shape)
    else:
        weight_matrix = None

    score, up_opt = tf.metrics.mean_iou(
        y_true, y_pred, n_classes, weights=weight_matrix, name="m1")

    session = KTF.get_session()
    if not iou_test:
        session.run(tf.local_variables_initializer())
        # running_vars = tf.get_collection(
        #     tf.GraphKeys.GLOBAL_VARIABLES, scope="mean_iou_met")
        # running_vars = [tf.get_variable(
        # "metrics/metric_mean_iou/mean_iou_met/total_confusion_matrix")]
        # print("vars", running_vars)
        # running_vars_initializer = tf.variables_initializer(
        #     var_list=running_vars)
        # session.run(running_vars_initializer)

        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)

    else:
        # this is only for test_iou/iou.py script
        running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="m1")
        # print("vars", running_vars)
        running_vars_initializer = tf.variables_initializer(
            var_list=running_vars)
        session.run(running_vars_initializer)
        session.run(up_opt)

    return score
