import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append([*args])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = list(map(np.asarray, zip(*batch)))[
            0].T  # FIXME: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
        states, actions, rewards, next_states, done = np.vstack(batch[0]), batch[1], np.vstack(batch[2]), batch[3], \
                                                      batch[4]
        return states, actions, rewards, next_states, done

    def sample_last(self):
        batch = self.memory[-1]
        return batch

    def __len__(self):
        return len(self.memory)
