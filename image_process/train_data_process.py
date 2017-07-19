import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def distort_color(image,color_ordering=0):
    if color_ordering==0 :
        image = tf.image.random_brightness(image,max_delta= 32. / 255)
        image = tf.image.random_saturation(image, lower= 0.5, upper= 1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower= 0.5, upper= 1.5)
    elif color_ordering==1 :
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    else :
        pass
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([[[0.0,0.0,1.0,1.0]]])

    if(image.dtype != tf.float32):
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)

    bbox_b,bbox_s,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_b,bbox_s)

    distorted_image = tf. image.resize_images(distorted_image,height,width,method=np.random.randint(4))

    distorted_image = tf.image.random_flip_left_right(distorted_image)

    distorted_image = distort_color(distorted_image, np.random.randint(2))

    return distorted_image

raw_data = tf.gfile.FastGFile("./input/DOL/img_00165.jpg").read()

with tf.Session() as sess:
    data = tf.image.decode_jpeg(raw_data)
    #print data.eval()
    print data.eval().shape
    plt.imshow(data.eval())
    plt.show()

    batched = tf.expand_dims(
        tf.image.convert_image_dtype(data,dtype=tf.float32),0
    )
    result=preprocess_for_train(data,720,1280,None)
    plt.imshow(result.eval())
    plt.show()