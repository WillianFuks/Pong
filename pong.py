import gym
import numpy as np
import tensorflow as tf
import glob
import os
import pickle
from typing import List, Tuple
from collections import deque


TOTAL_EPISODES = tf.constant(10000)
RELOAD_MODEL = True
TOTAL_ACTIONS = tf.constant(3)
FILE_PATH = './model.pckl'
H = tf.constant(200)  # Hidden neurons
GAMMA = tf.constant(.99, dtype=tf.float32)  # for reward discount factor
BETA = tf.constant(.9, dtype=tf.float32)  # decay factor for RMSProp weighted average updates
LAMBD = tf.constant(0, dtype=tf.float32)  # regularizer factor
ALPHA = tf.constant(1e-4, dtype=tf.float32)  # learning rate
EPSILON = tf.constant(1e-6, dtype=tf.float32)  # safety constant to avoid zero division
BATCH_SIZE = tf.constant(512)  # How many actions to run before running backpropagation
BATCH_SAVE = tf.constant(5)  # How many episodes to accumulate to save the model


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    )
)
def reward_go_to(rewards: tf.Tensor, gamma: float = GAMMA) -> tf.Tensor:
    """
    Implements discounted rewards as given by R_t = \sum^inf_k=0 \gamma^k r_{t+k}
    specifically designed for the Pong game, which means any time a reward is observed
    the summation over rewards is restarted.
    """
    size = tf.shape(rewards)[0]
    R = tf.TensorArray(tf.float32, size=size, clear_after_read=False)
    R = R.write(size - 1, rewards[size - 1])
    for t in tf.range(size - 2, -1, -1):  # -2 because R[-1] has already been processed
        reward = rewards[t]
        R = R.write(t, reward if reward != 0 else R.read(t + 1) * gamma + reward)
    R = R.stack()
    return R


def preprocess_img(x: tf.Tensor):
    """
    Preprocess 210x160x3 uint8 frame into 5600 (80x70) 2D (5600x1) float32 tensorflow vector.
    """
    x = x[35:195, 10:150][::2, ::2, 0]
    x = tf.where((x == 144) | (x == 109), 0, x)
    x = tf.where(x != 0, 1, x)
    return tf.cast(tf.reshape(x, (-1, 1)), tf.float32)


@tf.function
def compute_action(w1, w2, b1, b2, x):
    z_2 = w2 @ (tf.nn.relu(w1 @ x) + b1) + b2  # logits
    z_2 = tf.transpose(z_2)
    action = tf.squeeze(
        tf.random.categorical(logits=z_2, num_samples=1), axis=1)
    return z_2, action


@tf.function
def update_vars(s_dw1, s_dw2, s_db1, s_db2, dw1, dw2, db1, db2, w1, w2, b1, b2, ALPHA,
                BETA):
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


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='w1'),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='w2'),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name='rewards'),
        tf.TensorSpec(shape=[None], dtype=tf.int64, name='actions'),
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name='logits'),
        tf.TensorSpec(shape=[], dtype=tf.float32, name='GAMMA'),
        tf.TensorSpec(shape=[], dtype=tf.int32, name='TOTAL_ACTIONS'),
        tf.TensorSpec(shape=[], dtype=tf.float32, name='LAMBD'),
    )
)
def compute_J_theta(
    w1: tf.Variable,
    w2: tf.Variable,
    rewards: tf.Tensor,
    actions: tf.Tensor,
    logits: tf.Tensor,
    GAMMA: float,
    TOTAL_ACTIONS: int,
    LAMBD: float
) -> tf.Tensor:
    R = reward_go_to(rewards, GAMMA)
    R -= tf.reduce_mean(R)
    R /= tf.math.reduce_std(R)
    actions_mask = tf.one_hot(actions, TOTAL_ACTIONS)
    log_P = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)
    J_theta = tf.reduce_mean(R * log_P) + LAMBD * (tf.norm(w1) ** 2 + tf.norm(w2) ** 2)  # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id8
    return J_theta


def step_env(action: np.array) -> Tuple[np.array, np.array, np.array]:
    obs, reward, done, _ = env.step(action)
    return obs.astype(np.int32), np.array(reward, np.int32), np.array(done, np.int32)


@tf.function
def graph_step_env(action: tf.Tensor):
    obs, reward, done = tf.numpy_function(step_env, [action],
                                         [tf.int32, tf.int32, tf.int32])
    obs = preprocess_img(obs)
    return obs, reward, done


@tf.function
def run_optimization(env, w1, w2, b1, b2):
    current_x = preprocess_img(tf.constant(env.reset(), tf.int32))
    previous_x = tf.zeros_like(current_x)
    mean_x = current_x

    s_dw1 = tf.Variable(tf.zeros_initializer()(shape=w1.shape), dtype=tf.float32)
    s_db1 = tf.Variable(tf.zeros_initializer()(shape=b1.shape), dtype=tf.float32)
    s_dw2 = tf.Variable(tf.zeros_initializer()(shape=w2.shape), dtype=tf.float32)
    s_db2 = tf.Variable(tf.zeros_initializer()(shape=b2.shape), dtype=tf.float32)

    done = tf.constant(0, dtype=tf.int32)
    episode_reward = tf.constant(0, dtype=tf.int32)
    avg_r = tf.constant(0, dtype=tf.float32)
    weighted_avg_r = tf.constant(0, tf.float32)
    print_flag = tf.constant(False, dtype=tf.bool)

    for episode in tf.range(TOTAL_EPISODES):
        logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        with tf.GradientTape() as tape:
            while tf.constant(True, tf.bool):
                i = logits.size()
                z_2, action = compute_action(w1, w2, b1, b2, mean_x)
                logits = logits.write(i, z_2)
                actions = actions.write(i, action)
                # Add +1 so to map to valid actions: 1 (no-op), 2 (move up) and 3 (move down)
                current_x, reward, done = graph_step_env(action[0] + 1)
                # Take a weighted average so we can track movement; from where the ball is
                # coming to where it's going
                mean_x = (1.5 * current_x + previous_x) / 2.5
                previous_x = current_x
                rewards = rewards.write(i, reward)
                episode_reward += reward

                if tf.cast(done, tf.bool):
                    done = tf.constant(0, tf.int32)
                    print_flag = tf.constant(True, dtype=tf.bool)
                    avg_r += tf.cast(episode_reward, tf.float32)
                    print('HAHAHAHHAHA ', avg_r)
                    episode_reward = tf.constant(0, tf.int32)
                    current_x = preprocess_img(tf.constant(env.reset(), tf.int32))
                    previous_x = tf.zeros_like(current_x)
                    mean_x = tf.identity(current_x)
                    episode += 1
                    tf.print(f'episode: {episode}')
                    # If played all episodes but batch size still is not enough to run BP then
                    # force the loop to break
                    if episode > TOTAL_EPISODES:
                        break
                # Time for backpropagation. Leave the loop so to exit the tape context for
                # performance reasons. Play until the end of the round (when reward is != 0)
                if i >= BATCH_SIZE and reward != 0:
                    break
            J_theta = compute_J_theta(w1, w2, rewards.concat(), actions.concat(),
                                      logits.concat(), GAMMA, TOTAL_ACTIONS, LAMBD)

        dw1, dw2, db1, db2 = tape.gradient(J_theta, [w1, w2, b1, b2])
        update_vars(s_dw1, s_dw2, s_db1, s_db2, dw1, dw2, db1, db2, w1, w2, b1, b2, ALPHA,
                    BETA)

        if not (episode + 1) % BATCH_SAVE and print_flag:
            tf.print(f'This is J_theta: {J_theta}')
            tf.print(f'This is total plays: {rewards.shape[0]}')
            tf.print('This is dw1: ', dw1[0, :100])
            tf.print('This is s_dw1: ', s_dw1[0, :100])
            tf.print('This is dw2: ', dw2[0, :100])
            tf.print('This is s_dw2: ', s_dw2[0, :100])

            weighted_avg_r = (
                .9 * weighted_avg_r + .1 * avg_r / BATCH_SAVE if weighted_avg_r != 0
                else avg_r / BATCH_SAVE
            )
            tf.print(f'AVERAGE RETURN IS: {avg_r / BATCH_SAVE}')
            tf.print(f'WEIGHTED AVERAGE RETURN IS: {weighted_avg_r}')
            tf.print(f'w1 sum: {tf.reduce_sum(w1)}')
            tf.print(f'w2 sum: {tf.reduce_sum(w2)}')
            tf.print(f'b1 sum: {tf.reduce_sum(b1)}')
            tf.print(f'b2 sum: {tf.reduce_sum(b2)}\n')
            avg_r = tf.constant(0, tf.float32)
            print_flag = tf.constant(False, tf.bool)
            with open(FILE_PATH, 'wb') as f:
                pickle.dump([w1, w2, b1, b2], f)


if __name__ == '__main__':
    env = gym.make('ALE/Pong-v5')

    if RELOAD_MODEL and os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'rb') as f:
            w1, w2, b1, b2 = pickle.load(f)
    else:
        w1 = tf.Variable(tf.initializers.GlorotNormal()(shape=(H, current_x.shape[0])),
                         dtype=tf.float32)
        b1 = tf.Variable(tf.zeros_initializer()(shape=(w1.shape[0], 1)), dtype=tf.float32)
        w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(3, H)), dtype=tf.float32)
        b2 = tf.Variable(tf.zeros_initializer()(shape=(w2.shape[0], 1)), dtype=tf.float32)

    run_optimization(env, w1, w2, b1, b2)
