import time
import tensorflow as tf
import train_refactor
from tensorflow.examples.tutorials.mnist import input_data

eval_secs= 10


def evaluation(mnist,in_shape,out_shape):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=(None, in_shape), name="x_input")
        y_ = tf.placeholder(tf.float32, shape=(None, out_shape), name="y_input")

        val_feed={
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        y=train_refactor.inference(x,in_shape,out_shape,None)
        corr_pre= tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
        acc=tf.reduce_mean(tf.cast(corr_pre,tf.float32))

        variables_averages= tf.train.ExponentialMovingAverage(
            train_refactor.moving_decay
        )

        variables_to_restore= variables_averages.variables_to_restore()

        saver=tf.train.Saver(variables_to_restore)
        ckpt_m=None
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(
                    "./model_re"
                )
                if(ckpt_m==ckpt):
                    print("train finished")
                    return
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step= ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    acc_score=sess.run(acc,feed_dict=val_feed)
                    print ("Round %s acc : %g"%(global_step,acc_score))
                else:
                    print "no ckpt"
                    return
                time.sleep(eval_secs)
                ckpt_m=ckpt


def main(argv=None):
    minst = input_data.read_data_sets("./train", one_hot=True)
    Input_node = minst.train.images[0].shape[0]
    Output_node = minst.train.labels[0].shape[0]

    evaluation(minst, Input_node, Output_node)

if __name__=='__main__':

    tf.app.run()
