import numpy as np
import sys
import tensorflow as tf
from pympler import tracker


# w1 = tf.random.normal((3, 3))
# w2 = tf.random.normal((1, 3))
# b1 = tf.random.normal((3, 1))
# b2 = tf.random.normal((1, 1))
H = 200

w1 = tf.Variable(tf.initializers.GlorotNormal()(shape=(H, 5600)),
                 dtype=tf.float32)
b1 = tf.Variable(tf.zeros_initializer()(shape=(w1.shape[0], 1)), dtype=tf.float32)

w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(3, H)), dtype=tf.float32)
b2 = tf.Variable(tf.zeros_initializer()(shape=(w2.shape[0], 1)), dtype=tf.float32)

@tf.function
def compute_action(w1, w2, b1, b2, x):
    z_2 = w2 @ (tf.nn.relu(w1 @ x) + b1) + b2  # logits
    action = tf.squeeze(
        tf.random.categorical(logits=tf.transpose(z_2)[0:1, :], num_samples=1),
        axis=1
    )
    return z_2, action

def preprocess_img(x: np.array):
    """
    Preprocess 210x160x3 uint8 frame into 5600 (80x70) 2D (5600x1) float32 tensorflow vector.
    """
    x = x[35:195, 10:150]
    x = x[::2, ::2, 0]
    x[(x == 144) | (x == 109)] = 0
    x[x != 0] = 1
    return tf.convert_to_tensor(x.ravel()[..., np.newaxis], dtype=tf.float32)


x = tf.random.normal((5600, 1))
a = np.random.normal(size=(210, 160, 3))
tr = tracker.SummaryTracker()
preprocess_img(a)
# z_2, action = compute_action(w1, w2, b1, b2, x)
tr.print_diff()
sys.exit(1)

print(f'z_2: {z_2}')
print(f'action: {action}')

# with tf.GradientTape() as tape:
while True:
    z_2, action = compute_action(w1, w2, b1, b2, x)
    x = tf.random.normal((5600, 1))
