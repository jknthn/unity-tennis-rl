import os
import math
import numpy as np
import torch
from tensorboardX import SummaryWriter


def training_loop(env, brain_name, agent, config):
    scores = []
    avg_scores = []
    writer = SummaryWriter()
    last_max = -math.inf

    for e in range(1, config.training.episode_count):

        rewards = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations

        for t in range(config.training.max_t):
            rand = 0
            if e < 1000:
                rand = 1.0
            elif e < 2000:
                rand = 0.5
            action = agent.act(state, False, rand)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            rewards.append(reward)
            if any(done):
                break

        scores.append(sum(np.array(rewards).sum(1)))
        avg_scores.append(sum(scores[-100:]) / 100)
        writer.add_scalar('stats/reward', scores[-1], e)
        writer.add_scalar('stats/avg_reward', avg_scores[-1], e)

        print(f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | Best average: {max(avg_scores):10.4f} | Last score: {scores[-1]:10.4f}', end='\r')

        if e > 100 and avg_scores[-1] > last_max and ((avg_scores[-1] - last_max) > 0.05 or avg_scores[-1] > config.training.solve_score):
            for i, to_save in enumerate(agent.agents):
                torch.save(to_save.actor_local.state_dict(), os.getcwd() + f"/models/by_score/score_{avg_scores[-1]:.2f}_actor_{i}.weights")
                torch.save(to_save.critic_local.state_dict(), os.getcwd() + f"/models/by_score/score_{avg_scores[-1]:.2f}_critic_{i}.weights")
            last_max = avg_scores[-1]

        if e % 100 == 0:
            print(f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | Best average: {max(avg_scores):10.4f} | Last score: {scores[-1]:10.4f}')

        if avg_scores[-1] > config.training.solve_score and not config.training.continue_after_solve:
            break
