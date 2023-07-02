class MCAgent:
    
    def __init__(self, env, gamma, random_state=None):
        import numpy as np
        
        self.env = env
        self.gamma = gamma
        
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        
        self.states = range(self.num_states)
        self.actions = range(self.num_actions)
        
        self.Q = {s:np.zeros(self.num_actions) for s in self.states}
        self.V = {s:0 for s in self.states}
        
        
    def policy_evaluation(self, policy, threshold, max_iter=None, verbose=True):
        
        if max_iter is None:
            max_iter = float('inf')
            
        n = 0
        while n < max_iter:
            n += 1
            
            V_old = self.V.copy()
            self.V = {s:0 for s in self.states}
            
            for s in self.states:
                a = policy[s]
                transitions = self.env.P[s][a]
                
                for prob, next_state, reward, done in transitions:
                    G_estimate = reward + self.gamma * V_old[next_state]
                    self.V[s] += prob * G_estimate
                
            max_diff = max([abs(V_old[s] - self.V[s]) for s in self.states])
            if max_diff < threshold:
                break
            
        if verbose:
            print(f'Policy evaluation required {n} iterations to converge.')
            
        
if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    from envs import * 
    from utils import success_rate

    base_4x4_env = gym.make('FrozenLake-v1', render_mode='ansi')    
    env = base_4x4_env
    
    env.reset()
    print(env.render())
    
    mc = MCAgent(env, gamma=0.99)
    
    states = range(env.observation_space.n)
    direct_actions = [2, 2, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 0, 2, 2, 0]
    random_actions = [2, 0, 1, 3, 0, 0, 2, 0, 3, 1, 3, 0, 0, 2, 1, 0]
    careful_actions = [0, 3, 3, 3, 0, 0, 3, 0, 3, 1, 0, 0, 0, 2, 2, 0]
    direct_policy = {s:a for s, a in zip(states, direct_actions)}
    random_policy = {s:a for s, a in zip(states, random_actions)}
    careful_policy = {s:a for s, a in zip(states, careful_actions)}
        
    mc.policy_evaluation(random_policy, threshold=1e-6)
    print()
    
    frozen_lake_show_value(env, mc.V, digits=3)
    print()
    
    sr, info = success_rate(env, policy=random_policy, n=1000, max_steps=50)
    
    print(sr)
    print(info)