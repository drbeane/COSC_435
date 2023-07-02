def animate_episode(env, steps, sleep=0.1):
    from IPython.display import clear_output
    import matplotlib.pyplot as plt
    import time

    env.reset()

    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')
    plt.show()

    for i in range(steps):
        time.sleep(sleep)
        clear_output(wait=True)
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        plt.imshow(env.render(mode='rgb_array'))
        plt.axis('off')
        plt.show()

        if terminated:
            print('Terminated')
            break

            
def render_mp4(videopath):
    from base64 import b64encode
    mp4 = open(videopath, 'rb').read()
    base64_encoded_mp4 = b64encode(mp4).decode()
    return f'<video width=400 controls><source src="data:video/mp4;' \
           f'base64,{base64_encoded_mp4}" type="video/mp4"></video>'


def record_episode(
    env, policy, steps=250, env_seed=None, action_seed=None, 
    fps=30, updates=False, filename='temp', vec_env=False):

    from gym.wrappers import RecordVideo
    import os
    import shutil
    from IPython.display import display, HTML

    if os.path.exists('temp_videos'): shutil.rmtree('temp_videos')
    vid_env = RecordVideo(env, video_folder='temp_videos', name_prefix=filename, new_step_api=True)
    vid_env.metadata['render_fps'] = fps

    if env_seed is None:
        obs, info = vid_env.reset()
    else:
        obs, info = vid_env.reset(seed=env_seed)
        
    if action_seed is not None:
        vid_env.action_space.seed(action_seed)

    if updates:
        print(f'{"step":<6}{"action":<8}{"new state":<30}{"reward":<8}{"terminated":<12}{"info":<8}')


    for i in range(steps):
        if type(policy) == dict:
            a = policy[obs]
        else:
            a = policy(vid_env, obs)
        
        if vec_env:
            obs, reward, terminated, truncated, info = vid_env.step([a])
        else:
            obs, reward, terminated, truncated, info = vid_env.step(a)
        s = str(obs)

        if updates:
            print(f'{i:<6}{a:<8}{s:30}{reward:<8}{str(terminated):<12}{str(info):<8}')

        if terminated:
            break

    print(i + 1, 'steps completed.')

    vid_env.close()

    if not os.path.exists('videos'): os.mkdir('videos')
    shutil.move(f'/content/temp_videos/{filename}-episode-0.mp4', f'/content/videos/{filename}.mp4')
    shutil.rmtree('temp_videos')

    html = render_mp4(f'/content/videos/{filename}.mp4')
    display(HTML(html))


def random_action(env, obs):
    return env.action_space.sample()

'''
ToDo: Set some kind of seed. 
'''
def success_rate(env, policy, n, max_steps=1000):
    goal_count = 0
    total_eps = 0
    total_eps_s = 0
    total_eps_f = 0
    
    for i in range(n):
        obs, info = env.reset()

        for j in range(max_steps):
            try:
                a = policy[obs]
            except:
                a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            if terminated:
                break
        total_eps += j
        if env.unwrapped.s == env.observation_space.n - 1:
            goal_count += 1
            total_eps_s += j
        else:
            total_eps_f += j

    sr = goal_count / n
    info = {
        'sr' : sr,
        'avg_len' : total_eps / n,
        'avg_len_s' : total_eps_s / goal_count,
        'avg_len_f' : total_eps_f / (n - goal_count),
        
    }

    return sr, info