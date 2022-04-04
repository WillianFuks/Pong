import tensorflow as tf
from pong import reward_go_to, compute_action, compute_J_theta, LAMBD


def test_compute_action():
    w1 = tf.Variable(tf.eye(3, 3))
    b1 = tf.Variable(tf.constant([[1.]], dtype=tf.float32))
    b2 = tf.identity(b1)
    w2 = tf.Variable(tf.constant([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=tf.float32))
    tx = tf.constant([[1], [-2], [-3]], dtype=tf.float32)
    tz_2, taction = compute_action(w1, w2, b1, b2, tx)
    tf.debugging.assert_equal(tz_2, tf.constant([3, 1, 1], dtype=tf.float32))
    tf.debugging.assert_less_equal(taction, tf.constant([2], dtype=tf.int64))

    tx = tf.constant([[-2], [-3], [1]], dtype=tf.float32)
    w2.assign(tf.constant([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=tf.float32))
    tz_2, taction = compute_action(w1, w2, b1, b2, tx)
    tf.debugging.assert_equal(tz_2, tf.constant([1, 1, 3], dtype=tf.float32))
    tf.debugging.assert_less_equal(taction, tf.constant([2], dtype=tf.int64))


def test_reward_go_to():
    t = tf.TensorArray(dtype=tf.float32, size=3)
    t.write(0, 0)
    t.write(1, 0)
    t.write(2, 1)
    r = reward_go_to(t.concat(), gamma=tf.constant(0.5))
    assert list(reward_go_to(t.concat(), gamma=tf.constant(0.5)).numpy()) == [0.25, 0.5, 1]

    t = tf.TensorArray(dtype=tf.float32, size=6)
    t.write(0, 0.)
    t.write(1, 0)
    t.write(2, 1)
    t.write(3, 0)
    t.write(4, 0)
    t.write(5, -1)
    assert list(reward_go_to(t.concat(), gamma=tf.constant(0.5)).numpy()) == (
        [0.25,  0.5,  1., -.25, -.5 , -1.]
    )


def test_compute_J_theta():
    w1 = tf.ones((3, 3), dtype=tf.float32)
    w2 = tf.ones((1, 1), dtype=tf.float32)
    rewards = tf.constant([0., 0., 0., -1.], dtype=tf.float32)
    actions = tf.constant([0, 1, 2, 1], dtype=tf.int64)
    logits = tf.constant(
        [[1., 2., 3.], [0., 3., 4.], [3., 0., 5.], [1., -1, 0]], dtype=tf.float32
    )
    J_theta = compute_J_theta(w1, w2, rewards, actions, logits)

    R = reward_go_to(rewards)
    R -= tf.reduce_mean(R)
    R /= tf.math.reduce_std(R)
    mask = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=tf.float32)
    logs = tf.reduce_sum(mask * tf.nn.log_softmax(logits), axis=1)
    expected = tf.reduce_mean(R * logs) + LAMBD * (3 ** 2 + 1 ** 2)
    tf.debugging.assert_near(J_theta, expected, atol=1e-6)


# def test_compute_J_theta_derivative():
    # w1 = tf.Variable(tf.initializers.GlorotNormal()(shape=(3, 3)), dtype=tf.float32)
    # w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(3, 3)), dtype=tf.float32)
    # b1 = tf.Variable(tf.zeros((w1.shape[0], 1)))
    # b2 = tf.Variable(tf.zeros((w2.shape[0], 1)))
    # GAMMA = 0.5
    # TOTAL_ACTIONS = 3
    # LAMBD = 0.1
    # rewards = tf.constant([0., -1.], dtype=tf.float32)
    # actions = tf.constant([0, 1], dtype=tf.int64)
    # X = tf.constant([[1., 2.], [2., 3.], [3., 4.]], dtype=tf.float32)
    # with tf.GradientTape() as tape:
        # logits, _ = compute_action(w1, w2, b1, b2, X)
        # J_theta = compute_J_theta(w1, w2, rewards, actions, logits)
    # dw1, dw2, db1, db2 = tape.gradient(J_theta, [w1, w2, b1, b2])

    # # Test if for loop nudges the derivates out
    # logits = tf.TensorArray(tf.float32, 2)
    # with tf.GradientTape() as tape:
        # for i in range(2):
            # logits.write(i, compute_action(w1, w2, b1, b2, X[:, i][..., tf.newaxis])[0])
        # J_theta = compute_J_theta(w1, w2, rewards, actions, logits.concat())

    # dw1_2, dw2_2, db1_2, db2_2 = tape.gradient(J_theta, [w1, w2, b1, b2])
    # tf.debugging.assert_near(dw1, dw1_2)
    # tf.debugging.assert_near(dw2, dw2_2)
    # tf.debugging.assert_near(db1, db1_2)
#     tf.debugging.assert_near(db2, db2_2)
