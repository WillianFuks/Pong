import gym
import numpy as np
import tensorflow as tf
import glob
import os
import pickle
from typing import List
from collections import deque


TOTAL_EPISODES = 10000
RELOAD_MODEL = True
TOTAL_ACTIONS = 3
FILE_PATH = './model.pckl'
H = 200  # Hidden neurons
GAMMA = .99  # for reward discount factor
BETA = tf.constant(.9, dtype=tf.float32)  # decay factor for RMSProp weighted average updates
LAMBD = tf.constant(0, dtype=tf.float32)  # regularizer factor
ALPHA = tf.constant(1e-3, dtype=tf.float32)  # learning rate
EPSILON = tf.constant(1e-6, dtype=tf.float32)  # safety constant to avoid zero division
BATCH_SIZE = 512  # How many actions to run before running backpropagation
BATCH_SAVE = 50  # How many episodes to accumulate to save the model


def pong_reward_to_go(rewards: List[float], gamma: float = GAMMA) -> np.array:
    """
    Implements discounted rewards as given by R_t = \sum^inf_k=0 \gamma^k r_{t+k}
    specifically designed for the Pong game, which means any time a reward is observed
    the summation over rewards is restarted.
    """
    R = deque()
    R.append(rewards[-1])
    for t in range(len(rewards) - 2, -1, -1):  # -2 because R[-1] has already been processed
        reward = rewards[t]
        R.appendleft(reward if reward != 0 else R[0] * gamma + reward)
    return np.array(R)


assert list(pong_reward_to_go([0, 0, 1], gamma=0.5)) == [0.25, 0.5, 1]
assert list(pong_reward_to_go([0, 0, 1, 0, 0, -1], gamma=0.5)) == (
    [0.25,  0.5,  1., -.25, -.5 , -1.]
)


def preprocess_img(x: np.array):
    """
    Preprocess 210x160x3 uint8 frame into 5600 (80x70) 2D (5600x1) float32 tensorflow vector.
    """
    x = x[35:195, 10:150][::2, ::2, 0]
    x[(x == 144) | (x == 109)] = 0
    x[x != 0] = 1
    return tf.convert_to_tensor(x.ravel()[..., np.newaxis], dtype=tf.float32)


# @tf.function
def compute_action(w1, w2, b1, b2, x):
    z_2 = w2 @ (tf.nn.relu(w1 @ x) + b1) + b2  # logits
    action = tf.squeeze(
        tf.squeeze(tf.random.categorical(logits=tf.transpose(z_2), num_samples=1), axis=1),
        axis=1
    )
    return z_2, action

tw1 = tf.eye(3, 3)
tb1 = tf.constant([1.], dtype=tf.float32)
tb2 = tf.identity(tb1)
tw2 = tf.constant([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=tf.float32)
tx = tf.constant([[1], [-2], [-3]], dtype=tf.float32)
tz_2, taction = compute_action(tw1, tw2, tb1, tb2, tx)
tf.debugging.assert_equal(tz_2, tf.constant([[3], [0], [0]], dtype=tf.float32))
tf.debugging.assert_less_equal(taction, tf.constant([2], dtype=tf.int64))

tx = tf.constant([[-2], [-3], [1]], dtype=tf.float32)
tw2 = tf.constant([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=tf.float32)
tz_2, taction = compute_action(tw1, tw2, tb1, tb2, tx)
tf.debugging.assert_equal(tz_2, tf.constant([[0], [0], [3]], dtype=tf.float32))
tf.debugging.assert_less_equal(taction, tf.constant([2], dtype=tf.int64))


# @tf.function
def update_vars(dw1, dw2, db1, db2, w1, w2, b1, b2, s_dw1, s_db1, s_dw2, s_db2):
    # RMSProp weighted average update
    s_dw1.assign(BETA * s_dw1 + (1 - BETA) * (dw1 ** 2))
    s_db1.assign(BETA * s_db1 + (1 - BETA) * (db1 ** 2))
    s_dw2.assign(BETA * s_dw2 + (1 - BETA) * (dw2 ** 2))
    s_db2.assign(BETA * s_db2 + (1 - BETA) * (db2 ** 2))
    # Backprop
    w1.assign(w1 + ALPHA * dw1 / (tf.sqrt(s_dw1) + EPSILON))  # Addition because this is Gradient Ascend
    b1.assign(b1 + ALPHA * db1 / (tf.sqrt(s_db1) + EPSILON))
    w2.assign(w2 + ALPHA * dw2 / (tf.sqrt(s_dw2) + EPSILON))
    b2.assign(b2 + ALPHA * db2 / (tf.sqrt(s_db2) + EPSILON))


env = gym.make('ALE/Pong-v5')

current_x = preprocess_img(env.reset())
previous_x = tf.zeros_like(current_x)
mean_x = current_x

if RELOAD_MODEL and os.path.exists(FILE_PATH):
    with open(FILE_PATH, 'rb') as f:
        w1, w2, b1, b2 = pickle.load(f)
else:
    w1 = tf.Variable(tf.initializers.GlorotNormal()(shape=(H, current_x.shape[0])),
                     dtype=tf.float32)
    b1 = tf.Variable(tf.zeros_initializer()(shape=(w1.shape[0], 1)), dtype=tf.float32)
    w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(3, H)), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros_initializer()(shape=(w2.shape[0], 1)), dtype=tf.float32)

s_dw1 = tf.Variable(tf.zeros_initializer()(shape=w1.shape), dtype=tf.float32)
s_db1 = tf.Variable(tf.zeros_initializer()(shape=b1.shape), dtype=tf.float32)
s_dw2 = tf.Variable(tf.zeros_initializer()(shape=w2.shape), dtype=tf.float32)
s_db2 = tf.Variable(tf.zeros_initializer()(shape=b2.shape), dtype=tf.float32)

logits = tf.zeros([w2.shape[0], 0], dtype=tf.float32)
actions = []
done = False
rewards = []
episode_reward = 0
avg_r = 0
episode = 0
print_flag = False

while episode < TOTAL_EPISODES:
    with tf.GradientTape() as tape:
        while True:
            z_2, action = compute_action(w1, w2, b1, b2, mean_x)
            logits = tf.concat([logits, z_2], axis=1)
            action = action.numpy()
            actions.append(action)
            # Add +1 so to map to valid actions: 1 (no-op), 2 (move up) and 3 (move down)
            current_x, reward, done, _ = env.step(action + 1)
            # Winning is rare so we try to boost its reward
            # reward = 10. if reward > 0 else reward
            current_x = preprocess_img(current_x)
            # Take a weighted average so we can track movement; from where the ball is
            # coming to where it's going
            mean_x = (1.5 * current_x + previous_x) / 2.5
            previous_x = current_x
            rewards.append(reward)
            episode_reward += reward
            # Reset episode if necessary
            if done:
                done = False
                print_flag = True
                avg_r += episode_reward
                episode_reward = 0
                obs = env.reset()
                current_x = preprocess_img(obs.copy())
                previous_x = tf.zeros_like(current_x)
                mean_x = tf.identity(current_x)
                episode += 1
                print(f'episode: {episode}')
                # If played all episodes but batch size still is not enough to run BP then
                # force the loop to break
                if episode > TOTAL_EPISODES:
                    break
            # Time for backpropagation. Leave the loop so to exit the tape context for
            # performance reasons. Play until the end of the round (when reward is != 0)
            if len(rewards) >= BATCH_SIZE and reward != 0:
                break
        R = pong_reward_to_go(rewards, GAMMA)  # Outside of Graph mode
        # R -= R.mean()  # Normalization does seem to help in convergence
        # R /= R.std()
        actions_mask = tf.transpose(tf.one_hot(actions, TOTAL_ACTIONS))
        log_P = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=0)  # Each column represents a point 't'
        J_theta = (
            tf.reduce_mean(R * log_P) + LAMBD * (tf.norm(w1) ** 2 + tf.norm(w2) ** 2)
        )  # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id8

    dw1, dw2, db1, db2 = tape.gradient(J_theta, [w1, w2, b1, b2])
    update_vars(dw1, dw2, db1, db2, w1, w2, b1, b2, s_dw1, s_db1, s_dw2, s_db2)

    if not (episode + 1) % BATCH_SAVE and print_flag:
        print(f'This is J_theta: {J_theta}')
        print(f'This is total plays: {len(R)}')
        print('This is dw1: ', dw1[0, :100])
        print('This is s_dw1: ', s_dw1[0, :100])
        print('This is dw2: ', dw2[0, :100])
        print('This is s_dw2: ', s_dw2[0, :100])

        print(f'AVERAGE RETURN IS: {avg_r / BATCH_SAVE}')
        print(f'w1 sum: {tf.reduce_sum(w1)}')
        print(f'w2 sum: {tf.reduce_sum(w2)}')
        print(f'b1 sum: {tf.reduce_sum(b1)}')
        print(f'b2 sum: {tf.reduce_sum(b2)}\n')
        avg_r = 0
        print_flag = False
        with open(FILE_PATH, 'wb') as f:
            pickle.dump([w1, w2, b1, b2], f)

    # resets everything
    logits = tf.zeros([w2.shape[0], 0], dtype=tf.float32)
    actions = []
    rewards = []
