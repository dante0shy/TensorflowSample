import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size = 100
layers_node = 500

image_size = 28
num_channels = 1

conv1_deep = 32
conv1_size = 5

conv2_deep = 64
conv2_size = 5

fc_size = 512

learning_rate = 0.01
learning_rate_decay = 0.99
regularization_rate = 0.0001
moving_decay = 0.99
training_step = 5000




def inference(in_tensor,in_shape,out_shape,regularizer):
    print in_shape

    in_tensor= tf.reshape(in_tensor, [-1, 28, 28, 1])

    with tf.variable_scope('layer1-conv1') :
        conv1_weights = tf.get_variable(
            "weight",[conv1_size,conv1_size,num_channels,conv1_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        conv1_biases = tf.get_variable(
            "bias",[conv1_deep],initializer=tf.constant_initializer(0.0)
        )
        conv1=tf.nn.conv2d(
            in_tensor,
            conv1_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2') :
        conv2_weights = tf.get_variable(
            "weight",[conv2_size,conv2_size,conv1_deep,conv2_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias",[conv2_deep],initializer=tf.constant_initializer(0.0)
        )

        conv2=tf.nn.conv2d(
            pool1,
            conv2_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    print pool2

    pool1_shape = pool2.get_shape().as_list()
    print pool1_shape
    nodes = pool1_shape[1]*pool1_shape[2]*pool1_shape[3]
    if pool1_shape[0]!=None :
        reshape = tf.reshape(pool2,[pool1_shape[0],nodes])
    else:
        reshape = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight",[nodes,fc_size],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))

        fc1_biases = tf.get_variable(
            "bias",[fc_size],initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshape,fc1_weights)+fc1_biases)
        # if train:
        #     fc1 = tf.nn.dropout(fc1,0.5)
        fc1 = tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight",[fc_size,out_shape],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )

        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))

        fc2_biases = tf.get_variable(
            "bias",[out_shape],initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases

    return logit





def build_model(mnist,in_shape,out_shape):


    x = tf.placeholder(tf.float32, shape=(None, in_shape), name="x_input")
    y_ = tf.placeholder(tf.float32, shape=(None, out_shape), name="y_input")

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    y = inference(x,in_shape,out_shape,regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_decay, global_step
    )

    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )

    # variable_y=inference(x, None,w1,b1,w2,b2)

    cross_entopy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = y, logits = y_
    )

    cross_entopy_mean= tf.reduce_mean(cross_entopy)

    loss= cross_entopy_mean+tf.add_n(tf.get_collection("losses"))

    learning_rate_pro = tf.train.exponential_decay(
        learning_rate,
        global_step,
        mnist.train.num_examples,
        learning_rate_decay
    )

    train_step = tf.train.GradientDescentOptimizer(
        learning_rate_pro
    ).minimize(
        loss,global_step=global_step
    )

    # train_op = tf.group(train_step, variable_averages_op)
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op('train')


    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.arg_max(y,1),
                tf.arg_max(y_,1)
            ),
            tf.float32
        )
    )

    with tf.Session() as sess:
        tf.initialize_all_variables().run()


        for i in range(training_step):
            x_m, y_m = mnist.train.next_batch(batch_size)
           # x_m=np.reshape(x_m,(-1,28,28,1))

            _,loss_v,step=sess.run(
                [train_op,loss,global_step], feed_dict={
                    x: x_m,
                    y_: y_m
                }
            )
            if(i%500==0):
                print "Round %d, train loss: %g" % (i,loss_v)

                tf.train.Saver().save(sess,"./model/model.ckpt",global_step=global_step)
        # print sess.run(w1)
        # print sess.run(w2)
        tf.train.Saver().save(sess, "./model/model.ckpt", global_step=global_step)

def main(argv=None):
    minst = input_data.read_data_sets("./train", one_hot=True)
    Input_node = minst.train.images[0].shape[0]
    Output_node = minst.train.labels[0].shape[0]

    build_model(minst, Input_node, Output_node)

if __name__=='__main__':

    tf.app.run()






