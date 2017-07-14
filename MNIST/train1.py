import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data


batch_size = 100
layers_node = 500
learning_rate = 0.001
learning_rate_decay = 0.99
regularization_rate = 0.0001
moving_decay = 0.99
training_step = 3000


def inference(inputT,avg_calss,w1,b1,w2,b2):
    if avg_calss==None:
        layers=tf.nn.relu(tf.matmul(inputT,w1)+b1)
        return tf.matmul(layers,w2)+b2
    else:
        layers = tf.nn.relu(tf.matmul(inputT, avg_calss.average(w1)) + avg_calss.average(b1))
        return tf.matmul(layers,avg_calss.average(w2)) + avg_calss.average(b2)

def build_model(mnist,in_shape,out_shape):


    x = tf.placeholder(tf.float32, shape=(None, in_shape), name="x_input")
    y_ = tf.placeholder(tf.float32, shape=(None, out_shape), name="y_input")

    w1 = tf.Variable(tf.truncated_normal([in_shape,layers_node],stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1,shape=[layers_node]))


    w2 = tf.Variable(tf.truncated_normal([layers_node,out_shape],stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[out_shape]))

    y = inference(x,None,w1,b1,w2,b2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_decay, global_step
    )

    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )

    # variable_y=inference(x, None,w1,b1,w2,b2)
    variable_y=inference(x, variable_averages,w1,b1,w2,b2)


    cross_entopy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        y, tf.arg_max(y_,1)
    )

    cross_entopy_mean= tf.reduce_mean(cross_entopy)

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    regularization = regularizer(w1)+regularizer(w2)

    loss= cross_entopy+regularization

    learning_rate_pro = tf.train.exponential_decay(
        learning_rate,
        global_step,
        minst.train.num_examples,
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
                tf.arg_max(variable_y,1),
                tf.arg_max(y_,1)
            ),
            tf.float32
        )
    )

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {
            x : minst.validation.images,
            y_: minst.validation.labels
        }
        test_feed = {
            x : minst.test.images,
            y_: minst.test.labels
        }
        train_feed = {
            x : minst.train.images,
            y_: minst.train.labels
        }


        for i in range(training_step):
            if(i%500==0):
                loss_m_t = sess.run(
                    cross_entopy_mean,feed_dict=train_feed
                )
                loss_m_v = sess.run(
                    cross_entopy_mean,feed_dict=validate_feed
                )
                accuracy_v = sess.run(
                    accuracy,feed_dict=validate_feed
                )
                print "Round %d, train loss: %g, val loss: %g, val acc: %g" % (i,loss_m_t,loss_m_v,accuracy_v)
            # print sess.run(w1)
            # print i
            x_m,y_m = minst.train.next_batch(batch_size)
            sess.run(
                train_op,feed_dict= {
                    x : x_m,
                    y_ : y_m
                }
            )

        test_acc = sess.run(
            accuracy, feed_dict=test_feed
        )

        print "finial round %d, acc: %g"%(training_step,test_acc)
        tf.train.Saver().save(sess,"./model/model.ckpt")
        # print sess.run(w1)
        # print sess.run(w2)

if __name__=='__main__':
    minst = input_data.read_data_sets("./train", one_hot=True)
    Input_node=minst.train.images[0].shape[0]
    Output_node=minst.train.labels[0].shape[0]


    build_model(minst,Input_node,Output_node)







