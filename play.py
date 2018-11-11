

def play_loop(env, brain_name, agent, playthrougs=10):

    for e in range(playthrougs):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        done = False

        while not done:
            env_info = env.step(agent.act(state))[brain_name]
            state = env_info.vector_observations
            done = any(env_info.local_done)
