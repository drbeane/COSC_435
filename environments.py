import gymnasium as gym

class FrozenLakeMod(gym.Wrapper):

    def __init__(self, env, rew=[0,0,1]):
        super().__init__(env)
        self.num_steps = 0
        self.rew = rew

    def step(self, action):
        self.num_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            #reward = 100 if reward == 1 else -100
            reward = self.rew[2] if reward == 1 else self.rew[1]
        else:
            #reward = -10
            reward = self.rew[0]

        return obs, reward, terminated, truncated, info