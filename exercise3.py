import math
import os
import random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_dim, output_dim, action_space, hidden_dim=128):
        # self.batch_size is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 1000
        self.target_update_freq = 200
        self.t = 0
        self.tau = 0.005
        self.lr = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps_done = 0
        self.action_space = action_space

        self.n_actions = output_dim
        self.n_observations = input_dim

        self.policy_net = DQN(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=self.lr, amsgrad=True
        )
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.action_space.sample()]], device=self.device, dtype=torch.long
            )

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping (questionable)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.t += 1
        if self.t % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def plot_durations(episode_durations):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Result")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    plt.grid()
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig("plots/exercise3.png", bbox_inches="tight")


def main():
    episode_durations = []
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    dqn_agent = DQNAgent(len(state), env.action_space.n, env.action_space)
    device = dqn_agent.device
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 600

    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = dqn_agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Store the transition in memory
            dqn_agent.memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            dqn_agent.optimize_model()

            if done:
                episode_durations.append(t + 1)
                break

            # Move to the next state
            state = next_state

        if (
            # make sure the moving average is pretty close to 500
            [duration >= 499 for duration in episode_durations[-110:]].count(True)
            >= 100
            # and that the current episode shows ideal performance
            and episode_durations[-1] >= 499
            # and that we have enough episodes for the exercise
            and i_episode + 1 >= 500
        ):
            print(
                f"Environment solved in {i_episode + 1} episodes! Average duration: {sum(episode_durations[-100:]) / 100:.2f}"
            )
            break
    env.close()

    print("Complete")
    plot_durations(episode_durations)

    # save weights
    os.makedirs("weights", exist_ok=True)
    torch.save(dqn_agent.policy_net.state_dict(), "weights/exercise3_policy_net.pth")
    torch.save(dqn_agent.target_net.state_dict(), "weights/exercise3_target_net.pth")

    # create videos
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(
        env, "videos", episode_trigger=lambda x: True, name_prefix="exercise3"
    )
    state, info = env.reset()
    for i_episode in range(3):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = dqn_agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if done:
                break

            # Move to the next state
            state = torch.tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)

    env.close()
    print("Video saved to 'videos/exercise3*.mp4'")


if __name__ == "__main__":
    main()
