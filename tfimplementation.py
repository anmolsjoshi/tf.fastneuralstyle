import vgg_model
import model_lengstrom
import scipy.misc
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import time
import functools


########################################################################################################################

STYLE_IMAGE = 'styles_resized/' + sys.argv[1] + '_resized.jpg'

STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
CONTENT_LAYER = 'relu4_2'

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

CONTENT_WEIGHT = [5e0]
STYLE_WEIGHT = [1e2]
TV_WEIGHT = [1e-5]

LEARNING_RATE = 1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 2

SAVE_EVERY = 10000
PRINT_EVERY = 100

DATA_PATH = sys.argv[2] + '_resized'

########################################################################################################################

def gram_matrix(features):
    _, _, _, C = features.get_shape().as_list()
    matrix = tf.reshape(features, [-1, C])
    G = tf.matmul(tf.transpose(matrix), matrix)
    G = tf.divide(G, tf.to_float(tf.size(features)))
    return G

########################################################################################################################

def filenames_queue(path):
    names = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return names


def images(batch_names):
    batch = []
    for name in batch_names:
        batch.append(scipy.misc.imread(name, mode='RGB'))
    return batch

def get_image(image):
    img = np.asarray(scipy.misc.imread(image, mode='RGB'))
    img = np.expand_dims(img, axis=0)
    return img

########################################################################################################################

def main():

########################################################################################################################

    with tf.Graph().as_default() as graph:


        x_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 3))
        x_vgg_center = vgg_model.preprocess(x_images)

        y_images = model_lengstrom.style_transfer(image=x_images/255.0)
        y_vgg_center = vgg_model.preprocess(y_images)

        style_image = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
        style_vgg_center = vgg_model.preprocess(style_image)

        x_vgg = vgg_model.net(VGG_MODEL, x_vgg_center)
        y_vgg = vgg_model.net(VGG_MODEL, y_vgg_center)
        style_vgg = vgg_model.net(VGG_MODEL, style_vgg_center)

########################################################################################################################

        #content_loss = 0
        #for layer in CONTENT_LAYER:
        #    content_loss += tf.divide(2*tf.nn.l2_loss((y_vgg[layer] - x_vgg[layer])), BATCH_SIZE*tf.to_float(tf.size(x_vgg[layer])))

        content_loss = tf.reduce_sum(CONTENT_WEIGHT * tf.divide(2 * tf.nn.l2_loss(y_vgg[CONTENT_LAYER] - x_vgg[CONTENT_LAYER]),
                                                                BATCH_SIZE*tf.to_float(tf.size(x_vgg[CONTENT_LAYER]))))

        _, He, Wi, _ = y_images.get_shape().as_list()

        x_tv_loss = (tf.nn.l2_loss(y_images[:, 1:, :, :] - y_images[:, :He-1, :, :]))
        y_tv_loss = (tf.nn.l2_loss(y_images[:, :, 1:, :] - y_images[:, :, :Wi-1, :]))
        tv_y_size = tf.to_float(tf.size(y_images[:, 1:, :, :]))
        tv_x_size = tf.to_float(tf.size(y_images[:, :, 1:, :]))
        tv_loss = tf.reduce_sum(TV_WEIGHT * 2 * (x_tv_loss / tv_x_size + y_tv_loss / tv_y_size) / BATCH_SIZE)

        #style_loss = 0
        #for layer in STYLE_LAYERS:
        #    G = gram_matrix(y_vgg[layer])
        #    A = gram_matrix(style_vgg[layer])
        #    style_loss += tf.divide(2*tf.nn.l2_loss(G-A), BATCH_SIZE*tf.to_float(tf.size(G)))

        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = y_vgg[style_layer]

            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size

            s_layer = style_vgg[style_layer]
            bs, height, width, filters = map(lambda i: i.value, s_layer.get_shape())
            size = height * width * filters
            style_feats = tf.reshape(s_layer, (bs, height * width, filters))
            style_feats_T = tf.transpose(style_feats, perm=[0, 2, 1])
            style_gram = tf.matmul(style_feats_T, style_feats) / size

            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / size)

        style_loss = tf.reduce_sum(STYLE_WEIGHT * functools.reduce(tf.add, style_losses) / BATCH_SIZE)

        #loss = CONTENT_WEIGHT*content_loss + TV_WEIGHT*tv_loss + STYLE_WEIGHT*style_loss
        loss = content_loss + tv_loss + style_loss

        with tf.name_scope('weighted_losses'):
            tf.summary.scalar('weighted content loss', content_loss)
            tf.summary.scalar('weighted style loss', style_loss)
            tf.summary.scalar('weighted regularizer loss', tv_loss)
            tf.summary.scalar('total loss', loss)

        summary = tf.summary.merge_all()

        ########################################################################################################################

        global_step = tf.Variable(0, name="Global_Step", trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

########################################################################################################################

########################################################################################################################

        with tf.Session() as sess:

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter('summary/', sess.graph)
            sess.run(tf.global_variables_initializer())

            epoch = 0
            start_time = time.time()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            for epoch in (range(NUM_EPOCHS)):

                filenames = filenames_queue(DATA_PATH)
                num_iterations = len(filenames)//BATCH_SIZE

                for iter in (range(num_iterations)):
                    current = iter * BATCH_SIZE
                    end = current + BATCH_SIZE
                    batch_names = filenames[current:end]
                    batch = images(batch_names)
                    assert len(batch) == BATCH_SIZE
                    _, c_loss, s_loss, v_loss, t_loss, step = sess.run([train_op, content_loss, style_loss,
                                                                        tv_loss, loss, global_step],
                                                                       feed_dict={x_images:batch,
                                                                                  style_image:get_image(STYLE_IMAGE)})
                    if (iter+1)%PRINT_EVERY == 0:
                        print('Step %d Content Loss %.2e Style Loss %.2e Total Variation Loss %.2e Train Loss %.2e'
                              % (iter+1, c_loss, s_loss, v_loss, t_loss))

                    if (iter+1)%SAVE_EVERY == 0:
                        save_path = 'checkpoints/'+sys.argv[1]+'/Epoch'+str(epoch+1)+'/'+str(iter+1)+'/model.ckpt'
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saver.save(sess, save_path=save_path)
                        print ('---------------------------------SAVING CHECKPOINT OF MODEL at Iteration %d---------------------------------'%(iter+1))
                print ('---------------------------------COMPLETED EPOCH %d---------------------------------' % (epoch+1))
            save_path = 'models/'+sys.argv[1]+'/final/fast_neural_style'
            print ('-----------------------------------------TRAINING COMPLETE----------------------------------------')
            print ('---------------------------------SAVING FINAL CHECKPOINT OF MODEL---------------------------------')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver.save(sess, save_path)
            print ('---------------------------------TIME TAKEN = %.2f seconds --------------------------------------,' % (time.time()-start_time))

########################################################################################################################

if __name__ == '__main__':
    main()




