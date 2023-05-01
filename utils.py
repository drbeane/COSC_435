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
