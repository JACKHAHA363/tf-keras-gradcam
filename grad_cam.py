""" tf.keras model for heatmap
"""
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda
import numpy as np

def to_channel_last(channel_first_tensor):
    """ Transpose channel_first to channel_last
    """
    return tf.transpose(channel_first_tensor, perm=[0, 2, 3, 1])

def normalize(tensor):
    """ Normalize each feature map. Assuming tensor is channel last """
    mean, var = tf.nn.moments(tensor, axes=[1, 2, 3])
    return (tensor - mean[:, None, None, None]) / tf.sqrt(var[:, None, None, None])

def get_weighted_avg_fm(inputs):
    """ Computed weighted average of feature map. Weights are mean of gradient """
    target_fm = inputs[0]
    grad = inputs[1]
    weights = tf.reduce_mean(grad, axis=[1, 2])
    weighted_fm = target_fm * weights[:, None, None, :]
    return tf.reduce_mean(weighted_fm, axis=-1, keepdims=True)

def to_heatmap(weighted_avg_fm, img_width, img_height):
    """ Resize and make it from 0 to 1 """
    heatmap = tf.image.resize_bicubic(images=weighted_avg_fm, size=(img_width, img_height))
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap

def grad_cam(orig_model, layer_name):
    """
        :param orig_model: An instance of tf.keras.Model. The original CNN model
        :param layer_name: The name of target layer whose output is used to backprop
        :return: A new tf.keras.Model, whose output is concated with heatmap.
    """
    # make sure the original model only give prediction
    assert isinstance(orig_model, K.Model)
    assert len(orig_model.outputs) == 1
    assert len(orig_model.inputs) == 1
    is_channel_first = isinstance(model.layers[0], Conv2D) and model.layers[0].data_format == 'channels_first'
    img = model.input
    if is_channel_first:
        img = to_channel_last(img)
    img_width = img.shape[1].value
    img_height = img.shape[2].value

    # get output loss
    # sum of all sliced outputs according to predicted class
    output = orig_model.outputs[0]
    num_classes = output.shape[1].value
    pred = tf.argmax(output, axis=1)
    oh_pred = tf.one_hot(pred, num_classes)
    sum_of_class_outputs = tf.reduce_sum(output * oh_pred)

    # Get the feature map and gradients
    target_fm = [layer.output for layer in model.layers if layer.name == layer_name][0]
    grad_layer = K.layers.Lambda(lambda fm: K.backend.gradients(sum_of_class_outputs, target_fm)[0])
    grad = grad_layer(target_fm)

    # transpose to channel last
    # [bsz, width, height, n_channel]
    if is_channel_first:
        transpose_layer = Lambda(to_channel_last)
        target_fm = transpose_layer(target_fm)
        grad = transpose_layer(grad)

    # Normalize gradient per example
    grad = Lambda(normalize)(grad)

    # Weighted average featuremap
    weighted_avg_fm = Lambda(get_weighted_avg_fm)(inputs=[target_fm, grad])

    # From feature map to heatmap
    heatmap = Lambda(lambda weighted_avg_fm: to_heatmap(weighted_avg_fm,
                                                        img_width=img_width,
                                                        img_height=img_height))(weighted_avg_fm)

    # Construct a new model that output both prediction and avg_weighted_fm
    gcam_model = K.Model(inputs=orig_model.inputs, outputs=orig_model.outputs + [heatmap])
    return gcam_model

if __name__ == '__main__':
    model = K.applications.ResNet50(weights='imagenet')
    gcam_model = grad_cam(orig_model=model, layer_name='activation_48')

    # Test if is able to become a SavedModel.pb for serving
    sess = K.backend.get_session()
    tf.saved_model.simple_save(
            sess,
            'gcam_models',
            inputs={'input_image': gcam_model.input},
            outputs={t.name: t for t in gcam_model.outputs})

    # Test the result including gradient
    sess.run(tf.global_variables_initializer())
    dummy_input = np.zeros(shape=[2, 224, 224, 3], dtype=np.float32)
    outputs = sess.run(gcam_model.outputs, feed_dict={gcam_model.input: dummy_input})
    print(outputs[0].shape)  # (2, 1000)
    print(outputs[1].shape)  # (2, 7, 7, 1024)




