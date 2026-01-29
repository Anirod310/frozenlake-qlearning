import os 
import json
import gymnasium as gym
import numpy as np
from config import NUM_EPISODES, MAX_TRAIN_STEPS, LEARNING_RATE, GAMMA, EPSILON, IS_SLIPPERY, ENV_ID

train_env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY)

Q = np.zeros((train_env.observation_space.n, train_env.action_space.n))

successes = 0

for episode in range(NUM_EPISODES):

    state, info = train_env.reset()
    step = 0
    done = False

    while not done and step < MAX_TRAIN_STEPS :
        if np.random.uniform(0, 1) < EPSILON:
            action = train_env.action_space.sample() # generate a random action for exploration
        else :
            best_actions = np.flatnonzero(Q[state] == Q[state].max()) # get all actions with max value
            action = np.random.choice(best_actions) # randomly select one of the best actions
        
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

# ensure results folder exists then save Q and metrics
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

q_path = os.path.join(results_dir, "q_table.npy")
metrics_path = os.path.join(results_dir, "metrics.json")

try:
    np.save(q_path, Q)
    with open(metrics_path, 'w') as f:
        json.dump({"env_id": ENV_ID, "is_slippery": IS_SLIPPERY, "NUM_EPISODES": NUM_EPISODES, "successes": int(successes), "success_rate": successes/NUM_EPISODES}, f)
    print(f"Q table saved to {q_path}\n")
    print(f"Metrics saved to {metrics_path}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"Failed to save files to {results_dir}: {e}")
    raise 
