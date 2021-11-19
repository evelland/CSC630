import tensorflow as tf

import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

learning_rate = 0.015
training_iteration = 30
batch_size = 100
display_step = 2

x = tf.placeholder("float", [None, 784]) # input, 28 by 28 pixel image
y = tf.placeholder("float", [None, 10]) # output
weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
w_h = tf.histogram_summary("weights", weight)
b_h = tf.histogram_summary("biases", bias)

with tf.name_scope("weightx_bias") as scope:
        model = tf.nn.softmax(tf.matmul(x, weight) + bias)
with tf.name.scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))
    tf.scalar_summary("cost_function", cost_function)
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.initialize_all_variables()

merged_summary_op = tf.merge_all_summaries()

with tf.Session() as sess:
    sess.run(init)

    summary_writer = tf.train.SummaryWriter('')

    for iteration in range(training_iteration):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_exampes/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})/total_batch
            avg_cost += sess.run(merged_summary_op, feed_dict = {x: batch_xs, y: batch_ys})
            summary_str = sess.run(merged_summary_op, feed_dict = {x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration * total_batch + 1)
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Tuning completed!")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))