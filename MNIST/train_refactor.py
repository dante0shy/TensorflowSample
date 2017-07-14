import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data


batch_size = 100
layers_node = 500
learning_rate = 0.001
learning_rate_decay = 0.99
regularization_rate = 0.0001
moving_decay = 0.99
training_step = 5000

def get_weight_variable(shape, reqularizer):
    weights =  tf.get_variable(
        "weights",shape,initializer= tf.truncated_normal_initializer(stddev=0.1)
    )

    if(reqularizer!=None):
        tf.add_to_collection("losses", reqularizer(weights))
    return weights


def inference(in_tensor,in_shape,out_shape,regularizer):
    print in_shape
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [in_shape,layers_node],regularizer
        )
        biasses= tf.get_variable("biases",
                                 [layers_node],initializer=tf.constant_initializer(0.0)
                                 )
        layer1= tf.nn.relu(tf.matmul(in_tensor,weights)+biasses)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [layers_node,out_shape],regularizer
        )
        biasses = tf.get_variable("biases",
                                  [out_shape], initializer=tf.constant_initializer(0.0)
                                  )
        layer2 = tf.matmul(layer1, weights) + biasses
    return layer2

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
        y, tf.arg_max(y_,1)
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
            _,loss_v,step=sess.run(
                [train_op,loss,global_step], feed_dict={
                    x: x_m,
                    y_: y_m
                }
            )
            if(i%500==0):
                print "Round %d, train loss: %g" % (i,loss_v)

                tf.train.Saver().save(sess,"./model_re/model.ckpt",global_step=global_step)
        # print sess.run(w1)
        # print sess.run(w2)


def main(argv=None):
    minst = input_data.read_data_sets("./train", one_hot=True)
    Input_node = minst.train.images[0].shape[0]
    Output_node = minst.train.labels[0].shape[0]

    build_model(minst, Input_node, Output_node)

if __name__=='__main__':

    tf.app.run()






