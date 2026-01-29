import os
import json
import gymnasium as gym
import numpy as np
from config import NUM_EPISODES, MAX_TRAIN_STEPS, LEARNING_RATE, GAMMA, EPSILON  

Q = np.zeros((16, 4))  # 16 states, 4 actions

train_env = gym.make("FrozenLake-v1", is_slippery=False)

successes = 0

for episode in range(NUM_EPISODES):

    state, info = train_env.reset()
    step = 0
    done = False

    while not done and step < MAX_TRAIN_STEPS :
        if np.random.uniform(0, 1) < EPSILON:
            action = train_env.action_space.sample() # generate a random action for exploration
        else :
            best_actions = np.flatnonzero(Q[state] == Q[state].max())
            action = np.random.choice(best_actions)
        
        next_state, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        if reward == 1:
            successes += 1
        
        Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action]) 

        state = next_state
        step += 1

train_env.close()

print(f"Training with {NUM_EPISODES} episodes finished \n Successes during training : {successes} ({successes/NUM_EPISODES * 100:.2f} % success rate)\n\n Final Q table :")
print(f"{Q}\n")

# Save Q and metrics into results folder
q_path = os.path.join(os.path.dirname(__file__), "results", "q_table.npy")
metrics_path = os.path.join(os.path.dirname(__file__), "results", "metrics.json")
np.save(q_path, Q)
with open(metrics_path, 'w') as f:
    json.dump({"env_id": "FrozenLake-v1", "is_slippery": False, "NUM_EPISODES": NUM_EPISODES, "successes": int(successes), "success_rate": successes/NUM_EPISODES}, f)
print(f"Q table saved to {q_path}\n")
print(f"Metrics saved to {metrics_path}")












