import tensorflow as tf

print(tf.__version__)

x = tf.constant(0.)
y = tf.constant(1.)
for iteration in range(50):
    x = x + y
    y = y / 2
print(x.numpy())