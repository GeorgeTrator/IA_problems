# 1. Import necessary libraries
from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm # For progress bar
from matplotlib import pyplot as plt # For plotting

# 2. Define the Agent Class (BlackjackAgent)
class BlackjackAgent:
    """
    A Q-learning agent specifically for the Blackjack environment.
    """
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        # Q-table: Maps state -> array of Q-values for each action
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # To store the temporal difference errors during training
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.

        Args:
            obs: The current observation (player sum, dealer card, usable ace)

        Returns:
            The chosen action (0 for stand, 1 for hit).
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            # Choose the action with the highest Q-value for the current observation
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action using the Q-learning update rule."""
        # Estimate future Q-value: 0 if terminated, else max Q-value of next state
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # Calculate Temporal Difference (TD) error
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        # Update Q-value: Q(s,a) <- Q(s,a) + lr * TD_error
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        # Store the TD error for analysis
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Decreases the exploration rate (epsilon) over time."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

# 3. Define Visualization Helper Function
def get_moving_avgs(arr, window, convolution_mode="valid"):
    """Calculates moving averages for an array."""
    return np.convolve(
        np.array(arr).flatten(), # Ensure array is flat
        np.ones(window),         # Create a window of ones
        mode=convolution_mode    # 'valid' mode drops partial convolutions at ends
    ) / window                   # Normalize by window size

# 4. Set Hyperparameters
# Note: These are set for relatively quick training. For optimal policy,
# consider increasing n_episodes and decreasing learning_rate.
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
# Epsilon decay calculated to reach near final_epsilon around halfway through training
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# 5. Initialize Environment
# Use sab=False for the simpler version where natural Blackjack isn't special
env = gym.make("Blackjack-v1", sab=False)
# Wrap environment to automatically record episode statistics (reward, length)
# buffer_length ensures it stores stats for all episodes
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

# 6. Initialize Agent
agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# 7. Run the Training Loop
print(f"Starting training for {n_episodes} episodes...")
for episode in tqdm(range(n_episodes)): # tqdm adds a progress bar
    obs, info = env.reset() # Start new episode, get initial observation
    done = False

    # Play one episode step-by-step
    while not done:
        action = agent.get_action(obs) # Agent chooses action
        next_obs, reward, terminated, truncated, info = env.step(action) # Env executes action

        # Agent learns from the experience
        agent.update(obs, action, reward, terminated, next_obs)

        # Check if episode is finished (terminated or truncated)
        done = terminated or truncated
        obs = next_obs # Move to the next state

    # Decay epsilon after the episode is complete
    agent.decay_epsilon()

print("Training finished.")

# 8. Visualize Results
print("Generating plots...")
# Window size for moving average calculation
rolling_length = 500

fig, axs = plt.subplots(ncols=3, figsize=(18, 5)) # Increased figure size

# Plot smoothed Episode Rewards
axs[0].set_title("Episode Rewards (Smoothed)")
# Calculate moving average, using 'valid' mode trims ends where window doesn't fit
reward_moving_average = get_moving_avgs(
    env.return_queue, # Get rewards from the wrapper
    rolling_length,
    "valid"
)
axs[0].plot(range(rolling_length -1, n_episodes), reward_moving_average) # Adjust x-axis for 'valid' mode
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Average Reward")

# Plot smoothed Episode Lengths
axs[1].set_title("Episode Lengths (Smoothed)")
length_moving_average = get_moving_avgs(
    env.length_queue, # Get lengths from the wrapper
    rolling_length,
    "valid"
)
axs[1].plot(range(rolling_length -1, n_episodes), length_moving_average) # Adjust x-axis for 'valid' mode
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Average Length")

# Plot smoothed Training Error (Temporal Difference)
axs[2].set_title("Training Error (TD Error, Smoothed)")
# Use 'same' mode to keep output length same as input, averages edge points less accurately
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_xlabel("Training Step") # Note: x-axis is steps, not episodes here
axs[2].set_ylabel("Average TD Error")


plt.tight_layout() # Adjust plot spacing
plt.show() # Display the plots

# Close the environment (good practice, though often not strictly necessary for basic envs)
env.close()
print("Plots displayed and environment closed.")
