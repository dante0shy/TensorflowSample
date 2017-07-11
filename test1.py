import tensorflow as tf


# with tf.Session() as sess:
#     w1=tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))
#     w2=tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))
#     x=tf.constant([[0.5,0.6],[0.7,0.8]])
#     a=tf.matmul(x,w1)
#     y=tf.matmul(a,w2)
#     sess.run(w1.initializer)
#     sess.run(w2.initializer)
#     print sess.run(y)


w1=tf.Variable(tf.truncated_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))
x=tf.placeholder(tf.float32,shape=(3,2),name="input")
Y=tf.placeholder(tf.float32,shape=(3,1),name="input")


a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
cross_entopy=-tf.reduce_mean(y * tf.log(tf.clip_by_value(Y,1e-10,1.0)))
learning_rate= 0.001

train_step=tf.train.AdadeltaOptimizer(0.001).minimize(cross_entopy)

with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    print sess.run(y,feed_dict={x:[[0.1,0.2],[0.2,0.3],[0.3,0.4]]})