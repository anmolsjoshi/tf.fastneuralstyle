import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import model
import sys
import scipy.misc

model_path = sys.argv[2]

CONTENT_IMAGE = 'content_resized/' + sys.argv[1] + '_resized.jpg'

def get_image(image):
    img = np.asarray(scipy.misc.imread(image, mode='RGB'))
    img = np.expand_dims(img, axis=0)
    return img

def main():

    content_placeholder = tf.placeholder(tf.float32, shape=(1, 256, 256, 3))
    generated_images = model.style_transfer(content_placeholder)

    output_format = tf.cast(generated_images, tf.uint8)
    jpegs = tf.map_fn(lambda image: tf.image.encode_jpeg(image), output_format, dtype=tf.string)

    with tf.Session() as sess:
        file = tf.train.latest_checkpoint(model_path)
        if not file:
            print('Could not find trained model in %s' % model_path)
            return
        print('Using model from %s' % file)
        saver = tf.train.Saver()
        saver.restore(sess, file)

        images_t = sess.run(jpegs, feed_dict={content_placeholder:get_image(CONTENT_IMAGE)})
        with open('res.jpg', 'wb') as f:
            f.write(images_t[0])

if __name__ == '__main__':
   main()
