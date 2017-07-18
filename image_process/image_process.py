import matplotlib.pyplot as plt
import tensorflow as tf

raw_data = tf.gfile.FastGFile("./input/DOL/img_00165.jpg").read()

with tf.Session() as sess:
    data = tf.image.decode_jpeg(raw_data)
    #print data.eval()
    print data.eval().shape
    plt.imshow(data.eval())
    plt.show()

    # data= tf.image.convert_image_dtype(data,dtype=tf.float32)
    # print data.eval()
    # data = tf.image.encode_jpeg(data)
    # print data.eval()

    # resize_data_1=tf.image.resize_images(data,300,300,method=0)
    # print resize_data_1.get_shape()
    # plt.imshow(resize_data_1.eval())
    # plt.show()

    # resize_data_2= tf.image.central_crop(data,0.5)
    # plt.imshow(resize_data_2.eval())
    # plt.show()

    batched = tf.expand_dims(
        tf.image.convert_image_dtype(data,dtype=tf.float32),0
    )
    result=tf.image.draw_bounding_boxes(batched,[[[0.4,0.4,0.6,0.6]]])
    plt.imshow(result.eval().reshape(data.eval().shape))
    plt.show()