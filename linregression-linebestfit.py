
import tensorflow as tf
import numpy as np

#seed the random num generator if we want same results
np.random.seed(1)
# Create 100 random data
x_data = np.atleast_2d(np.random.rand(100).astype('float32')).T
y_data = x_data * 0.2 + 0.5

w = tf.Variable(tf.random_uniform([1, 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
model = tf.matmul(x, w) + b

n = 500 # number of iterations
loss = tf.reduce_mean(tf.square(y - model))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

# initialising variables
init = tf.global_variables_initializer()

# running the sesh
sess = tf.Session()
sess.run(init)

# training to fit the line
for step in range(n+1):   
    display_step = 20
    # Evaluate loss, weight and bias for each step
    # print out as we go
    _, loss_val, w_, b_ = sess.run([train, loss, w, b], 
             feed_dict={x: x_data, y: y_data})
    if step % display_step == 0:
        print('Epoch', step, 'completed out of', n, 'loss', loss_val)