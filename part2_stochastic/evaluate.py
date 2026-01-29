import gymnasium as gym
import numpy as np
from config import MAX_EVAL_STEPS, ENV_ID, IS_SLIPPERY, N_EVAL_EPISODES

Q = np.load("results/q_table.npy")
action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

eval_env = gym.make(ENV_ID, is_slippery=IS_SLIPPERY)

successes_eval = 0

for episode in range(N_EVAL_EPISODES):
    state, info = eval_env.reset()
    done = False
    total_reward = 0
    step = 0

    #print("Trajectory:")

    while not done and step < MAX_EVAL_STEPS:
        best_actions = np.flatnonzero(Q[state] == Q[state].max())
        action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        #print(f"step={step:02d}  state={state:02d}  action={action_names[action]:5s}  -> next_state={next_state:02d}  reward={reward}")

        total_reward += reward
        state = next_state
        step += 1
    
    if total_reward == 1:
        successes_eval += 1

eval_env.close()
print(f"Success rate (slippery=True) over {N_EVAL_EPISODES} episodes: {successes_eval}/{N_EVAL_EPISODES} = {successes_eval/N_EVAL_EPISODES:.2%}")