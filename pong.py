import gym
import numpy as np
import tensorflow as tf
import os
import pickle
from typing import List, Tuple


# tf.config.run_functions_eagerly(True)


TOTAL_EPISODES = 10000
RELOAD_MODEL = True
TOTAL_ACTIONS = 3
FILE_PATH = './model.pckl'
H = 200  # Hidden neurons
GAMMA = tf.constant(.99, dtype=tf.float32)  # for reward discount factor
BETA = tf.constant(.9, dtype=tf.float32)  # decay factor for RMSProp weighted average updates
LAMBD = tf.constant(0, dtype=tf.float32)  # regularizer factor
ALPHA = tf.constant(1e-3, dtype=tf.float32)  # learning rate
EPSILON = tf.constant(1e-12, dtype=tf.float32)  # safety constant to avoid zero division
BATCH_SIZE = 512  # How many actions to run before running backpropagation
BATCH_SAVE = 10  # How many episodes to accumulate to save the model


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.float32),
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
    R = tf.TensorArray(tf.float32, size=size)
    acc_reward = rewards[size - 1]
    R = R.write(size - 1, acc_reward)
    for t in tf.range(size - 2, -1, -1):  # -2 because R[-1] has already been processed
        acc_reward = acc_reward * gamma if not tf.cast(rewards[t], tf.bool) else rewards[t]
        R = R.write(t, acc_reward)
    R = R.stack()
    return R


def preprocess_img(x: np.array):
    """
    Preprocess 210x160x3 uint8 frame into 5600 (80x70) 2D (5600x1) float32 tensorflow vector.
    """
    x = x[35:195, 10:150][::2, ::2, 0]
    x[(x == 144) | (x == 109)] = 0
    x[x != 0] = 1
    return tf.convert_to_tensor(x.ravel()[..., np.newaxis], dtype=tf.float32)


@tf.function
def compute_action(x):
    z_2 = tf.transpose(w2 @ (tf.nn.relu(w1 @ x + b1)) + b2)  # logits
    action = tf.squeeze(
        tf.random.categorical(logits=z_2, num_samples=1), axis=1)
    return z_2, action


@tf.function
def update_vars(dw1, dw2, db1, db2):
    # RMSProp weighted average update
    s_dw1.assign(BETA * s_dw1 + (1 - BETA) * (dw1 ** 2))
    s_db1.assign(BETA * s_db1 + (1 - BETA) * (db1 ** 2))
    s_dw2.assign(BETA * s_dw2 + (1 - BETA) * (dw2 ** 2))
    s_db2.assign(BETA * s_db2 + (1 - BETA) * (db2 ** 2))
    # Backprop
    w1.assign_add(ALPHA * dw1 / (tf.sqrt(s_dw1) + EPSILON))  # Addition because this is Gradient Ascend
    b1.assign_add(ALPHA * db1 / (tf.sqrt(s_db1) + EPSILON))
    w2.assign_add(ALPHA * dw2 / (tf.sqrt(s_dw2) + EPSILON))
    b2.assign_add(ALPHA * db2 / (tf.sqrt(s_db2) + EPSILON))


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.int64),
        tf.TensorSpec(shape=[None, TOTAL_ACTIONS], dtype=tf.float32),
    )
)
def compute_J_theta(rewards: tf.Tensor, actions: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    R = reward_go_to(rewards)
    R -= tf.reduce_mean(R)
    R /= tf.math.reduce_std(R)
    actions_mask = tf.one_hot(actions, TOTAL_ACTIONS)
    log_P = tf.reduce_sum(actions_mask * tf.nn.log_softmax(logits), axis=1)
    J_theta = tf.reduce_mean(R * log_P) + LAMBD * (tf.norm(w1) ** 2 + tf.norm(w2) ** 2)  # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id8
    return J_theta


def step_env(env, action: np.array) -> Tuple[tf.Tensor, tf.Tensor, bool]:
    obs, reward, done, _ = env.step(action)
    return preprocess_img(obs), tf.constant(reward, tf.float32), done


if __name__ == '__main__':
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
        w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(TOTAL_ACTIONS, H)), dtype=tf.float32)
        b2 = tf.Variable(tf.zeros_initializer()(shape=(w2.shape[0], 1)), dtype=tf.float32)

    s_dw1 = tf.Variable(tf.zeros_initializer()(shape=w1.shape), dtype=tf.float32)
    s_db1 = tf.Variable(tf.zeros_initializer()(shape=b1.shape), dtype=tf.float32)
    s_dw2 = tf.Variable(tf.zeros_initializer()(shape=w2.shape), dtype=tf.float32)
    s_db2 = tf.Variable(tf.zeros_initializer()(shape=b2.shape), dtype=tf.float32)

    done = False
    episode_reward = 0
    avg_r = 0
    weighted_avg_r = 0
    episode = 0
    print_flag = False

    while episode < TOTAL_EPISODES:
        logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        with tf.GradientTape() as tape:
            while True:
                i = logits.size()
                z_2, action = compute_action(mean_x)
                logits = logits.write(i, z_2)
                actions = actions.write(i, action)
                # Add +1 so to map to valid actions: 1 (no-op), 2 (move up) and 3 (move down)
                current_x, reward, done = step_env(env, action.numpy()[0] + 1)
                # Take a weighted average so we can track movement; from where the ball is
                # coming to where it's going
                mean_x = (1.5 * current_x + previous_x) / 2.5
                previous_x = tf.identity(current_x)
                rewards = rewards.write(i, reward)
                episode_reward += int(reward)

                if done:
                    done = False
                    print_flag = True
                    avg_r += episode_reward
                    episode_reward = 0
                    current_x = preprocess_img(env.reset())
                    previous_x = tf.zeros_like(current_x)
                    mean_x = tf.identity(current_x)
                    episode += 1
                    print(f'episode: {episode}')
                # Time for backpropagation. Leave the loop so to exit the tape context for
                # performance reasons. Play until the end of the round (when reward is != 0)
                if i >= BATCH_SIZE and reward != 0:
                    break
            J_theta = compute_J_theta(rewards.concat(), actions.concat(), logits.concat())

        dw1, dw2, db1, db2 = tape.gradient(J_theta, [w1, w2, b1, b2])
        update_vars(dw1, dw2, db1, db2)

        # print(len(compute_action.get_concrete_function().graph.as_graph_def().node))

        if not (episode + 1) % BATCH_SAVE and print_flag:
            print(f'This is J_theta: {J_theta}')
            print(f'This is total plays: {rewards.size()}')
            print('This is dw1: ', dw1[0, :100])
            print('This is s_dw1: ', s_dw1[0, :100])
            print('This is dw2: ', dw2[0, :100])
            print('This is s_dw2: ', s_dw2[0, :100])

            weighted_avg_r = (
                .9 * weighted_avg_r + .1 * avg_r / BATCH_SAVE if weighted_avg_r != 0
                else avg_r / BATCH_SAVE
            )
            print(f'AVERAGE RETURN IS: {avg_r / BATCH_SAVE}')
            print(f'WEIGHTED AVERAGE RETURN IS: {weighted_avg_r}')
            print(f'w1 sum: {tf.reduce_sum(w1)}')
            print(f'w2 sum: {tf.reduce_sum(w2)}')
            print(f'b1 sum: {tf.reduce_sum(b1)}')
            print(f'b2 sum: {tf.reduce_sum(b2)}\n')
            avg_r = 0
            print_flag = False
            with open(FILE_PATH, 'wb') as f:
                pickle.dump([w1, w2, b1, b2], f)
