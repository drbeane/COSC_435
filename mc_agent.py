class MCAgent:
    
    def __init__(self, env, gamma, random_state=None):
        import numpy as np
        
        self.env = env
        self.gamma = gamma
        
        self.V = {}
        self.Q = {}
        
        # Store random state. This will be used for all methods as well. 
        self.random_state = random_state
        self.np_random_state = None
        if random_state is not None:
            np_state = np.random.get_state()               # Get current np random state
            np.random.seed(random_state)                   # Set seed to value provided
            self.np_random_state = np.random.get_state()   # Record new np random state
            np.random.set_state(np_state)                  # Reset old np random state       
        
        
        
    def control(self, episodes, epsilon, alpha, max_steps=None, show_progress=True):
        
        num_actions = self.env.action_space.n
        actions = range(num_actions)
        
        
if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    from envs import FrozenLakeMod

    env = gym.make(
        'FrozenLake-v1',
        desc=generate_random_map(size=10, p=0.9, seed=1),
        max_episode_steps=1000,
        render_mode='ansi'
    )
    
    mod_env = FrozenLakeMod(env, rew=[-1,-100, 100])

    env.reset()
    print(env.render())
    
    