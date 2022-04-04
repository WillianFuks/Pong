import gym
import numpy as np
import tensorflow as tf
import os
import pickle


# tf.config.run_functions_eagerly(True)

TOTAL_EPISODES = 20000
RELOAD_MODEL = True
TOTAL_ACTIONS = 3
FILE_PATH = './model.pckl'
H = 600  # Hidden neurons
GAMMA = .99  # for reward discount factor
BETA = .9  # decay factor for RMSProp weighted average updates
LAMBD = 1e-14  # regularizer factor
ALPHA = 1e-3  # learning rate
EPSILON = 1e-12  # safety constant to avoid zero division
BATCH_SIZE = 512  # How many actions to run before running backpropagation
BATCH_SAVE = 10  # How many episodes to run to save the model


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

@tf.function
def preprocess_img(x: tf.Tensor):
    """
    Preprocess 210x160x3 uint8 frame into 5600 (80x70) 2D (5600x1) float32 tensorflow vector.
    """
    x = x[35:195, 10:150][::2, ::2, 0]
    x = tf.where((x == 144) | (x == 109), 0, x)
    x = tf.where(x != 0, 1, x)
    return tf.cast(tf.reshape(x, (-1, 1)), tf.float32)


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
    # https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id8
    # https://www.coursera.org/lecture/ml-classification/learning-l2-regularized-logistic-regression-with-gradient-ascent-4JxyQ
    J_theta = tf.reduce_mean(R * log_P) - LAMBD * (tf.norm(w1) ** 2 + tf.norm(w2) ** 2)
    return J_theta


def step_env(action: np.array):
    obs, reward, done, _ = env.step(action)
    new_obs = env.reset().astype(np.int32) if done else np.empty(obs.shape, dtype=np.int32)
    return (
        obs.astype(np.int32),
        np.array(reward, np.int32),
        np.array(done, np.int32),
        new_obs
    )


@tf.function
def graph_step_env(action: tf.Tensor):
    obs, reward, done, reset_obs = tf.numpy_function(
        step_env,
        [action],
        [tf.int32, tf.int32, tf.int32, tf.int32]
    )
    done = tf.cast(done, tf.bool)
    reward = tf.cast(reward, tf.float32)
    obs = preprocess_img(obs)
    if done:
        reset_obs = preprocess_img(reset_obs)
    else:
        reset_obs = tf.cast(reset_obs, tf.float32)
    return obs, reward, done, reset_obs


@tf.function
def run_train_step(initial_obs: tf.Tensor, episode: tf.Tensor):
    obs_shape = initial_obs.shape
    previous_obs = tf.zeros_like(initial_obs)
    mean_obs = initial_obs

    logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
    rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    with tf.GradientTape() as tape:
        while tf.constant(True, tf.bool):
            i = logits.size()
            z_2, action = compute_action(mean_obs)
            logits = logits.write(i, z_2)
            actions = actions.write(i, action)
            # Add +1 so to map to valid actions: 1 (no-op), 2 (move up) and 3 (move down)
            obs, reward, done, new_obs = graph_step_env(action[0] + 1)
            # Take a weighted average so we can track movement; from where the ball is
            # coming to where it's going
            mean_obs = (1.5 * obs + previous_obs) / 2.5
            # mean_obs starts the loop with shape (5600, 1) as it's created in egar mode
            # but here it loses the shape as it's graph mode. By forcing the shape to
            # remain constant the while loop can process through.
            mean_obs.set_shape(obs_shape)
            previous_obs = obs
            previous_obs.set_shape(obs_shape)
            rewards = rewards.write(i, reward[..., tf.newaxis])
            episode_reward.assign_add(reward)
            if done:
                sum_reward.assign_add(episode_reward)
                episode_reward.assign(0.0)
                obs = new_obs
                mean_obs = obs
                mean_obs.set_shape(obs_shape)
                previous_obs = tf.zeros_like(obs)
                previous_obs.set_shape(obs_shape)
                episode.assign_add(1)
                tf.print('episode: ', episode)
            # Time for backpropagation. Leave the loop so to exit the tape context for
            # performance reasons. Play until the end of the round (when reward is != 0)
            if i >= BATCH_SIZE and reward != 0:
                break
        J_theta = compute_J_theta(rewards.concat(), actions.concat(), logits.concat())
    dw1, dw2, db1, db2 = tape.gradient(J_theta, [w1, w2, b1, b2])
    update_vars(dw1, dw2, db1, db2)
    return mean_obs


if __name__ == '__main__':
    env = gym.make('ALE/Pong-v5')
    obs = preprocess_img(tf.constant(env.reset(), tf.int32))

    if RELOAD_MODEL and os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'rb') as f:
            w1, w2, b1, b2 = pickle.load(f)
    else:
        w1 = tf.Variable(tf.initializers.GlorotNormal()(shape=(H, obs.shape[0])),
                         dtype=tf.float32)
        b1 = tf.Variable(tf.zeros_initializer()(shape=(w1.shape[0], 1)), dtype=tf.float32)
        w2 = tf.Variable(tf.initializers.GlorotNormal()(shape=(TOTAL_ACTIONS, H)),
                         dtype=tf.float32)
        b2 = tf.Variable(tf.zeros_initializer()(shape=(w2.shape[0], 1)), dtype=tf.float32)

    s_dw1 = tf.Variable(tf.zeros_initializer()(shape=w1.shape), dtype=tf.float32)
    s_db1 = tf.Variable(tf.zeros_initializer()(shape=b1.shape), dtype=tf.float32)
    s_dw2 = tf.Variable(tf.zeros_initializer()(shape=w2.shape), dtype=tf.float32)
    s_db2 = tf.Variable(tf.zeros_initializer()(shape=b2.shape), dtype=tf.float32)

    episode_reward = tf.Variable(0.0, tf.float32)
    episode = tf.Variable(1, tf.int32)
    new_ep_tmp = 0
    weighted_r = 0
    sum_reward = tf.Variable(0.0, tf.float32)

    while int(episode) < TOTAL_EPISODES:
        obs = run_train_step(obs, episode)

        # Use new_ep_tmp to detect it's a new episode and therefore needs to print.
        # Without this logic the code below would process several times.
        if not episode % BATCH_SAVE and int(new_ep_tmp) != int(episode):
            new_ep_tmp = int(episode)
            print(f'AVERAGE RETURN IS: {sum_reward / BATCH_SAVE}')
            print(f'w1 sum: {tf.reduce_sum(w1)}')
            print(f'w1 norm: {tf.norm(w1)}')
            print(f'w2 sum: {tf.reduce_sum(w2)}')
            print(f'b1 sum: {tf.reduce_sum(b1)}')
            print(f'b2 sum: {tf.reduce_sum(b2)}')
            weighted_r = float(
                0.9 * weighted_r + 0.1 * sum_reward / BATCH_SAVE if weighted_r else
                sum_reward / BATCH_SAVE
            )
            print('weighted return: ', weighted_r, '\n')
            sum_reward.assign(0.0, tf.float32)
            with open(FILE_PATH, 'wb') as f:
                pickle.dump([w1, w2, b1, b2], f)
