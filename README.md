# Pong

This is a re-implementation of the original Karpathy's [Atari-Pong](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) script ported to TensorFlow 2.

The approach here is a bit different from Karpathy's; while it still uses Policy Gradients, the AI trains through Gradient Ascent where cost function is the expected reward for running the current policy. Please refer to the [spinningup docs](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#id8) from OpenAI for a thorough discussion on details.

Here's an example of the AI playing after training for roughly 15k episodes; green player is the trained AI:

<p align="center">
  <img src="./pong.gif"
</p>
