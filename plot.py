import pickle
from PIL import Image, ImageSequence

import gym
import tensorflow as tf

from pong import compute_action, preprocess_img


FILE_PATH = './model.pckl'
IMG_FILE_PATH = 'pong.gif'

with open(FILE_PATH, 'rb') as f:
    w1, w2, b1, b2 = pickle.load(f)

done = False
images = []
env = gym.make('ALE/Pong-v5')
obs = preprocess_img(tf.constant(env.reset(), tf.int32))
previous_obs = tf.zeros_like(obs)
mean_obs = obs

while not done:
    img = env.render(mode='rgb_array')
    img = Image.fromarray(img)
    img = img.resize((300, 410))
    images.append(img)

    z_2, _ = compute_action(w1, w2, b1, b2, mean_obs)
    action = tf.argmax(z_2, axis=1)

    obs, _, done, _ = env.step(action[0].numpy() + 1)
    obs = preprocess_img(tf.constant(obs, tf.int32))

    mean_obs = (1.5 * obs + previous_obs) / 2.5
    previous_obs = obs

env.close()
images[0].save(IMG_FILE_PATH, save_all=True, append_images=images[1:], loop=0)
