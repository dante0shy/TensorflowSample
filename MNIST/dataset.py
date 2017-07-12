from tensorflow.examples.tutorials.mnist import input_data

minst= input_data.read_data_sets("./train",one_hot=True)

print minst.train.num_examples
print minst.test.num_examples
print minst.validation.num_examples
print minst.train.images[0].shape
print minst.train.labels[0].shape