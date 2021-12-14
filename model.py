import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mse, Huber, MeanSquaredError
import argparse
import numpy as np
import random
from buffer import ReplayBuffer, Transition
from proportional import Experience as PriorityReplayBuffer
import datetime

tf.keras.backend.set_floatx('float64')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class ActionValueFunctionApproximator:
    def __init__(self, state_dim, aciton_dim, args):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.loss_obj = tf.keras.metrics.Mean(name='train_loss')
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.update_step = 0
        self.model = self.create_model(args.layers_size[args.num_layers])

    @tf.autograph.experimental.do_not_convert
    def create_model(self, hidden_dims):
        _input = Input((self.state_dim,))
        layers = [Dense(dim, activation='relu', kernel_initializer='he_normal') for dim in
                  hidden_dims]
        output = Dense(self.action_dim)
        model = tf.keras.Sequential([_input, *layers, output])
        loss = Huber()
        # loss = MeanSquaredError()
        model.compile(loss=loss, optimizer=RMSprop(self.lr, rho=0.95))  # epsilon = 0.0001 Adam(self.lr)
        model.summary()
        return model

    def predict(self, state):
        # apply model predict method
        return self.model.predict(state)


class Agent:
    def __init__(self, args, env):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.model = ActionValueFunctionApproximator(self.state_dim, self.action_dim, args)
        self.target_model = ActionValueFunctionApproximator(self.state_dim, self.action_dim, args)
        self.target_update_mode = args.target_update_mode
        # represent the update frequency of the agent
        self.tau = args.tau
        self.target_update()
        self.batch_size = args.batch_size
        if args.buffer_type == 'uniform':
            self.buffer = ReplayBuffer(capacity=args.buffer_capacity)  # generate an instance of the replay buffer
        elif args.buffer_type == 'priority':
            self.buffer = PriorityReplayBuffer({'size': args.buffer_capacity, 'batch_size': self.batch_size})
        self.gamma = args.gamma
        # current exploration value
        self.epsilon = args.eps
        # Decay rate
        self.eps_decay = args.eps_decay
        # represents the low exploration bound
        self.eps_min = args.eps_min
        self.model_type = args.model_type
        self.buffer_type = args.buffer_type
        self.n_updates = 0
        self.eps_sigma = 650

    def update_epsilon(self):
        def gaussian_fun(x, sigma):
            return np.exp(-(x / sigma) ** 2)

        # self.epsilon *= self.eps_decay
        self.epsilon = gaussian_fun(self.n_updates, self.eps_sigma)
        self.n_updates += 1
        self.epsilon = max(self.epsilon, self.eps_min)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = np.reshape(state, [1, self.state_dim])
        q_value = self.model.predict(state)[0]
        return np.argmax(q_value)

    def target_update(self):
        if self.target_update_mode == 'hard':
            weights = self.model.model.get_weights()
            self.target_model.model.set_weights(weights)
        elif self.target_update_mode == 'soft':
            weights = np.array(self.model.model.get_weights())
            target_weights = np.array(self.target_model.model.get_weights())
            weights = self.tau * weights + (1 - self.tau) * target_weights
            self.target_model.model.set_weights(weights)
        else:
            raise Exception('Invalid update rule')

    def push(self, state, action, reward, next_state, done):
        transition = [state, action, reward, next_state, done]
        self.buffer.push(transition)

    def train_step(self, time_step):
        if self.buffer_type == 'uniform':
            states, actions, next_states, rewards, done = self.buffer.sample(batch_size=self.batch_size)
        elif self.buffer_type == 'priority':
            out, w, indices = self.buffer.sample(global_step=time_step, batch_size=self.batch_size)
            states, actions, next_states, rewards, done = out
        else:
            raise Exception
        actions = actions.astype(np.int)
        next_q_target_values = self.target_model.predict(next_states)
        pred = next_q_target_values.copy().max(axis=1)
        if self.model_type == 'DQN':
            next_q_target_values = next_q_target_values.max(axis=1)
        elif self.model_type == 'DDQN':
            next_actions = self.model.predict(next_states).argmax(axis=1)
            next_q_target_values = next_q_target_values[range(self.batch_size), next_actions]
        else:
            raise Exception
        Q_targets = rewards + (1 - done) * next_q_target_values * self.gamma
        Q_curr = self.model.predict(states)
        if self.buffer_type == 'uniform':
            Q_curr[range(self.batch_size), actions] = Q_targets
            loss = self.model.model.fit(states, Q_curr, batch_size=self.batch_size, verbose=0)

        elif self.buffer_type == 'priority':
            abs_td_errors = np.abs(pred - Q_curr[range(self.batch_size), actions]).tolist()
            Q_curr[range(self.batch_size), actions] = Q_targets

            self.buffer.update_priority(indices, abs_td_errors)
            loss = self.model.model.fit(states, Q_curr, sample_weight=w, batch_size=self.batch_size, verbose=0)

        else:
            raise Exception

        # Calculate the next Q values

        return loss

    def save(self, name):
        self.model.model.save(name)

