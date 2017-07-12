import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data


batch_size = 100
layers_node = 500
learning_rate = 0.001
learning_rate_decay = 0.99
regularization_rate = 0.0001
moving_decay = 0.99



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
    layer_dim=[2,10,10,10,1]
    n_layers=len(layer_dim)
    in_dim=layer_dim[0]

    # w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    # w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    # a = tf.matmul(x, w1)
    # y = tf.matmul(a, w2)

    cur_layer=x

    for i in range(1,n_layers):
        out_dim=layer_dim[i]

        weight=get_weight([in_dim,out_dim],0.001)
        bias = tf.Variable(tf.constant(0.1,shape=[out_dim]))
        cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
        in_dim=layer_dim[i]

    cross_entopy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(cur_layer, 1e-10, 1.0)))
    learning_rate = 0.001

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entopy)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #print sess.run(y, feed_dict={x: [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]})
        # print sess.run(w1)
        # print sess.run(w2)
        STEP=300

        for i in range(STEP):
            start= (i* batch_size)% num
            end = min(start+batch_size,num)

            mid=sess.run( train_step,feed_dict={x:x1[start:end],y_:y1[start:end]})

            if(i%10==0):
                total_cross_entropy= sess.run(
                    cross_entopy,feed_dict={x:x1,y_:y1}
                )
                print mid
                print ("round %d, loss: %g"%(i,total_cross_entropy))
                #print (sess.run(y,feed_dict={x:x1,y_:y1}))
        # print sess.run(w1)
        # print sess.run(w2)

if __name__=='__main__':
    minst = input_data.read_data_sets("./train", one_hot=True)
    Input_node=minst.train.images[0].shape[0]
    Output_node=minst.train.labels[0].shape[0]


    #build_model(1000,x,y)







