import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gym
import numpy as np
import time
import collections
from collections import namedtuple
from itertools import count

from model import Agent
import box
import datetime
from functools import partial
import tensorflow as tf
import argparse

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))

env = gym.make('CartPole-v1').unwrapped

# set up matplotlib
# ============================================================================
# Parameters

training_params = {
    'batch_size': 256,
    'gamma': 0.95,
    'eps': 1.0,
    'eps_min': 0.01,
    'target_update_mode': 'hard',  # use 'soft' or 'hard'
    'tau': 0.01,  # relevant for soft update
    'target_update_period': 50,  # relevant for hard update
    'buffer_capacity': 100000,
    'lr': 0.00025,

    'num_layers': 3,
    'layers_size': {3: [8, 8, 4], 5: [8, 8, 4, 4, 4]},
    'model_type': 'DQN',
    'buffer_type': 'uniform',
}
# [16, 16, 8, 8, 4] [8, 8, 4]
# training_params = {
#     'batch_size': 256,
#     'gamma': 0.99,
#     'eps': 1.1,
#     'eps_min': 0.05,
#     'eps_decay': 0.995,
#     'target_update_mode': 'hard',  # use 'soft' or 'hard'
#     'tau': 0.01,  # relevant for soft update
#     'target_update_period': 20,  # relevant for hard update
#     'grad_clip': 0.1, # didn't use
#     'buffer_capacity': 100000,
#     'lr': 0.001, # adam default!
#
#     'num_layers': 5,
#     'hidden_dim': 128,
#     'model_type': 'DQN',
#     'buffer_type': 'uniform',
# }

params = box.Box(training_params)
base_log_path = os.path.join(os.getcwd(), "logs/fit/tensorboard")


# ============================================================================

def run_exp(max_episodes, args, exp_name='', exp_dir=''):
    # Build neural network
    agent = Agent(args, env)
    summary_writer_train = tf.summary.create_file_writer(
        os.path.join(base_log_path, exp_dir,
                     exp_name + '_' + datetime.datetime.now().strftime("%d%m%Y-%H%M")))
    # Training loop
    max_score = 500
    task_score = 0
    # performances plots
    all_scores = []

    time_step = 0
    start_training_step = 1000
    obs_lim = env.observation_space.high
    # train for max_episodes
    for i_episode in range(max_episodes):
        agent.update_epsilon()
        with summary_writer_train.as_default():
            tf.summary.scalar('Epsilon', agent.epsilon, step=i_episode)
        ep_loss = []
        # Initialize the environment and state
        state = env.reset()
        score = 0
        for t in count():
            # Select and perform an action
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            if (1 in args.reward_shaping):  # better to stay centered and with a straight-up pole
                reward -= 0.5 * (np.abs(next_state[2]) / obs_lim[2])
                reward -= 0.5 * (np.abs(next_state[0]) / obs_lim[0])
            if (2 in args.reward_shaping) and done and t != max_score:
                reward = -100

            # Store the transition in memory
            agent.push(state, action, next_state, reward, done)

            # Update state
            state = next_state

            loss = 0
            # Perform one optimization step (on the policy network)
            if time_step > start_training_step:
                history = agent.train_step(time_step)
                loss = history.history['loss'][0]
            with summary_writer_train.as_default():
                tf.summary.scalar('Loss', loss, step=time_step)
            time_step += 1

            # soft target update
            if args.target_update_mode == 'soft':
                agent.target_update()

            if done or t >= max_score:
                print("Episode: {} | Current target score {} | Score: {}".format(i_episode + 1, task_score, score))
                break

        if args.target_update_mode == 'hard' and i_episode % args.target_update_period == 0:
            # update every args.target_update_period episodes the target network
            agent.target_update()

        # Save
        all_scores.append(score)

        # update task score
        if min(all_scores[-5:]) > task_score:
            task_score = min(all_scores[-5:])
            agent.save(agent.model.log_dir)

        with summary_writer_train.as_default():
            tf.summary.scalar('Score', score, step=i_episode)
            mean_score_last_100 = np.mean(all_scores[-100:])
            tf.summary.scalar('Avg_Score', mean_score_last_100, step=i_episode)
            tf.summary.scalar('Task_Score', task_score, step=i_episode)
        if args.early_stop and mean_score_last_100 >= 475:
            agent.save(agent.model.log_dir)
            break
    return mean_score_last_100


def scan_best_param(sweeps, algofunc, default_params, param_vec, param_name, file_name, param_ind):
    results = np.zeros((len(param_vec), sweeps))
    for sweep in range(sweeps):
        print(f'Sweep Index {sweep}')
        for idx, param in enumerate(param_vec):

            print(rf'{param_name} Value {param}')
            if param_ind == 0:
                default_params['gamma'] = param
                exp_name = f'Gamma_{param}'
            elif param_ind == 1:
                default_params['lr'] = param
                exp_name = f'lr_{param}'
            elif param_ind == 2:
                default_params['eps_decay'] = param
                exp_name = f'epsDecay_{param}'
            elif param_ind == 3:
                default_params['buffer_capacity'] = param
                exp_name = f'Capacity_{param}'
            elif param_ind == 4:
                default_params['batch_size'] = param
                exp_name = f'BatchSize_{param}'
            else:
                raise Exception
            params = box.Box(default_params)
            avg_score = algofunc(args=params, exp_name=exp_name, exp_dir=file_name)
            results[idx, sweep] = avg_score
    mean_results = results.mean(axis=1)
    return mean_results.max(), param_vec[mean_results.argmax()]


def dqn_sweeps(model_type='DQN', model_layers=3, update_type='hard', param_sweeps=[], sample_type='uniform'):
    default_params = training_params.copy()
    default_params['num_layers'] = model_layers
    default_params['target_update_mode'] = update_type
    default_params['model_type'] = model_type
    default_params['buffer_type'] = sample_type

    n_sweeps = 1
    # sweeps gamma
    if 'gamma' in param_sweeps:
        avg_score, gamma = scan_best_param(n_sweeps, partial(run_exp, max_episodes=default_params['n_episodes']),
                                           default_params,
                                           param_vec=[0.95, 0.99, 1], param_name=r'$\gamma$',
                                           file_name=f'{model_type}/{model_layers}_layers/{update_type}_update/{sample_type}_buffer',
                                           param_ind=0)
        default_params['gamma'] = gamma
    # # sweeps alpha
    if 'lr' in param_sweeps:
        avg_score, lr = scan_best_param(n_sweeps, partial(run_exp, max_episodes=default_params['n_episodes']),
                                        default_params,
                                        param_vec=[0.00025, 0.0001, 0.00005], param_name='lr',
                                        file_name=f'{model_type}/{model_layers}_layers/{update_type}_update/{sample_type}_buffer',
                                        param_ind=1)
        default_params['lr'] = lr
    # # sweeps epsilon
    if 'eps_decay' in param_sweeps:
        avg_score, decay = scan_best_param(n_sweeps, partial(run_exp, max_episodes=default_params['n_episodes']),
                                           default_params,
                                           param_vec=[0.995, 0.999, 0.99], param_name=r'$\epsilon_D$',
                                           file_name=f'{model_type}/{model_layers}_layers/{update_type}_update/{sample_type}_buffer',
                                           param_ind=2)
        default_params['eps_decay'] = decay
    # sweeps buffer capacity
    if 'capacity' in param_sweeps:
        avg_score, cap = scan_best_param(n_sweeps, partial(run_exp, max_episodes=default_params['n_episodes']),
                                         default_params,
                                         param_vec=[100000, 10000, 5000], param_name='Capacity',
                                         file_name=f'{model_type}/{model_layers}_layers/{update_type}_update/{sample_type}_buffer',
                                         param_ind=3)
        default_params['buffer_capacity'] = cap
    # sweeps batch size
    if 'batch_size' in param_sweeps:
        avg_score, batch_size = scan_best_param(n_sweeps, partial(run_exp, max_episodes=default_params['n_episodes']),
                                                default_params,
                                                param_vec=[256, 512, 128, 64], param_name='BatchSize',
                                                file_name=f'{model_type}/{model_layers}_layers/{update_type}_update/{sample_type}_buffer',
                                                param_ind=4)
        default_params['batch_size'] = batch_size

    path = os.path.join(os.getcwd(), 'outputs', 'training_params',
                        f'{model_type}_{model_layers}_layers_{update_type}_update_{param_sweeps}.csv')
    dict_to_csv(default_params, path)


def dict_to_csv(data, file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(file_name, 'w') as f:
        for key in data.keys():
            f.write("%s,%s\n" % (key, data[key]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Q', type=float, nargs='+', default=[2])
    parser.add_argument('--model_type', type=str, choices=['DQN', 'DDQN'], default='DQN')
    parser.add_argument('--model_layers', type=int, choices=[3, 5], default=3)
    parser.add_argument('--update_type', type=str, choices=['hard', 'soft'], default='hard')
    parser.add_argument('--param_sweep', type=str, nargs='+',
                        choices=['gamma', 'lr', 'eps_decay', 'capacity', 'batch_size'], default=[])
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--buffer_type', type=str, choices=['uniform', 'priority'], default='uniform')
    parser.add_argument('--exp', type=str, default='No_Name')
    parser.add_argument('--n_episodes', type=int, default=5000)
    parser.add_argument('--no_breaks', action='store_true', default=False)
    parser.add_argument('--reward_shaping', type=int, nargs='+', choices=[1, 2], default=[1])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    training_params['num_layers'] = args.model_layers
    training_params['target_update_mode'] = args.update_type
    training_params['model_type'] = args.model_type
    training_params['buffer_type'] = args.buffer_type
    training_params['early_stop'] = not args.no_breaks
    training_params['reward_shaping'] = args.reward_shaping
    n_episodes = args.n_episodes
    training_params['n_episodes'] = n_episodes

    if 2 in args.Q:
        params = box.Box(training_params)
        dict_to_csv(training_params, os.path.join(base_log_path, args.exp, args.exp + '.csv'))
        run_exp(n_episodes, args=params, exp_name=args.exp, exp_dir=args.exp)
    if 2.1 in args.Q:
        dqn_sweeps(model_type=args.model_type, model_layers=args.model_layers, update_type=args.update_type,
                   param_sweeps=args.param_sweep, sample_type=args.buffer_type)
    if 3 in args.Q:
        # Setting best parameters.
        training_params['reward_shaping'] = [1]
        training_params['buffer_type'] = 'priority'
        params = box.Box(training_params)
        dict_to_csv(training_params, os.path.join(base_log_path, args.exp, args.exp + '.csv'))
        run_exp(n_episodes, args=params, exp_name=args.exp, exp_dir=args.exp)


