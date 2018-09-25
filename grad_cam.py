""" tf.keras model for heatmap
"""
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import numpy as np

def _to_channel_last(channel_first_tensor):
    """ Transpose channel_first to channel_last
    """
    return tf.transpose(channel_first_tensor, perm=[0, 2, 3, 1])

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
        img = _to_channel_last(img)
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
    #if is_channel_first:
    #    target_fm = _to_channel_last(target_fm)
    #    grad = _to_channel_last(grad)

    # Normalize gradient per example
    #mean, var = tf.nn.moments(grad, axes=[1, 2, 3])
    #grad = (grad - mean) / tf.sqrt(var)

    # Compute weighted feature map
    # [bsz, width, height, n_channel
    #weights = tf.reduce_mean(grad, axis=[1, 2])
    #weighted_fm = target_fm * weights[:, None, None, :]
    #avg_weighted_fm = tf.reduce_mean(weighted_fm, axis=-1, keepdims=True)

    ## Resize weighted_feature map to img size
    ## Make it from 0 to 1
    #heatmap = tf.image.resize_bicubic(images=avg_weighted_fm, size=(img_width, img_height))
    #heatmap = tf.maximum(heatmap, 0)
    #heatmap = heatmap / tf.reduce_max(heatmap)

    # Construct a new model that output both prediction and avg_weighted_fm
    gcam_model = K.Model(inputs=orig_model.inputs, outputs=orig_model.outputs + [grad])
    return gcam_model

if __name__ == '__main__':
    #model = K.applications.ResNet50(weights='imagenet')
    #gcam_model = grad_cam(orig_model=model, layer_name='activation_48')

    model = K.Sequential()
    model.add(Conv2D(2, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=(28, 28, 3)))
    model.add(Conv2D(4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    gcam_model = grad_cam(orig_model=model, layer_name='max_pooling2d')

    # Test if is able to become a SavedModel.pb for serving
    """
    ╰─➤  saved_model_cli show --dir ./gcam_models --all

        MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

        signature_def['serving_default']:
            The given SavedModel SignatureDef contains the following input(s):
            inputs['input_image'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 28, 28, 3)
                name: conv2d_input:0
        The given SavedModel SignatureDef contains the following output(s):
            outputs['dense_1/Softmax:0'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 10)
                name: dense_1/Softmax:0
            outputs['lambda/gradients/flatten/Reshape_grad/Reshape:0'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 12, 12, 4)
                name: lambda/gradients/flatten/Reshape_grad/Reshape:0
        Method name is: tensorflow/serving/predict
    """
    sess = K.backend.get_session()
    tf.saved_model.simple_save(
            sess,
            'gcam_models',
            inputs={'input_image': gcam_model.input},
            outputs={t.name: t for t in gcam_model.outputs})

    # Test the result including gradient
    sess.run(tf.global_variables_initializer())
    dummy_input = np.zeros(shape=[2, 28, 28, 3], dtype=np.float32)
    outputs = sess.run(gcam_model.outputs, feed_dict={gcam_model.input: dummy_input})
    print(outputs[0].shape)  # (2, 10)
    print(outputs[1].shape)  # (2, 12, 12, 4)




