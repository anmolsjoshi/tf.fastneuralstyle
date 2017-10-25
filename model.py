import tensorflow as tf

def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('CONV') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='WEIGHT')
        convolved = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='CONV')

        normalized = batch_norm(convolved, output_filters)

        return normalized

def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('CONV_TRANSPOSE') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='WEIGHT')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], padding=padding, name='CONV_TRANSPOSE')

        normalized = batch_norm(convolved, output_filters)
        return normalized

def batch_norm(x, size):
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    beta = tf.Variable(tf.zeros([size]), name='BETA')
    scale = tf.Variable(tf.ones([size]), name='SCALE')
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='BATCH_NORM')

def residual(x, filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('RESIDUAL') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides, padding=padding)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides, padding=padding)

        residual = x + conv2

        return residual

def style_transfer(image):
    with tf.variable_scope('CONV1'):
        conv1 = tf.nn.relu(conv2d(image, 3, 32, 9, 1))
    with tf.variable_scope('CONV2'):
        conv2 = tf.nn.relu(conv2d(conv1, 32, 64, 3, 2))
    with tf.variable_scope('CONV3'):
        conv3 = tf.nn.relu(conv2d(conv2, 64, 128, 3, 2))
    with tf.variable_scope('RES1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('RES2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('RES3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('RES4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('RES5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('DECONV1'):
        deconv1 = tf.nn.relu(conv2d_transpose(res5, 128, 64, 3, 2))
    with tf.variable_scope('DECONV2'):
        deconv2 = tf.nn.relu(conv2d_transpose(deconv1, 64, 32, 3, 2))
    with tf.variable_scope('DECONV3'):
        deconv3 = tf.nn.tanh(conv2d_transpose(deconv2, 32, 3, 9, 1))

    generated_images = deconv3*127.5 + 255./2

    return generated_images