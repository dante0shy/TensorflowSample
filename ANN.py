import tensorflow as tf
from numpy.random import RandomState
# with tf.Session() as sess:
#     w1=tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))
#     w2=tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))
#     x=tf.constant([[0.5,0.6],[0.7,0.8]])
#     a=tf.matmul(x,w1)
#     y=tf.matmul(a,w2)
#     sess.run(w1.initializer)
#     sess.run(w2.initializer)
#     print sess.run(y)

batch_size = 8

def build_model(num,x1,y1):

    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
    x = tf.placeholder(tf.float32, shape=(None, 2), name="x_input")
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y_input")

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    cross_entopy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    learning_rate = 0.001

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entopy)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #print sess.run(y, feed_dict={x: [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]})
        print sess.run(w1)
        print sess.run(w2)
        STEP=1000

        for i in range(STEP):
            start= (i* batch_size)% num
            end = min(start+batch_size,num)

            mid=sess.run( train_step,feed_dict={x:x1[start:end],y_:y1[start:end]})

            if(i%100==0):
                total_cross_entropy= sess.run(
                    cross_entopy,feed_dict={x:x1,y_:y1}
                )
                print mid
                print ("round %d, loss: %g"%(i,total_cross_entropy))
                #print (sess.run(y,feed_dict={x:x1,y_:y1}))
        print sess.run(w1)
        print sess.run(w2)

if __name__=='__main__':
    rdm=RandomState(1)
    x=rdm.rand(1000,2)
    y=[[int(10*x1+10*x2)%2] for (x1,x2) in x]
    print x
    print y
    build_model(1000,x,y)







