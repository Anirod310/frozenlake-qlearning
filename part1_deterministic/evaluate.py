import gymnasium as gym 
import numpy as np
from config import MAX_EVAL_STEPS
from train import Q

action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

eval_env = gym.make("FrozenLake-v1", is_slippery = False)
state, info = eval_env.reset()

done = False
step = 0
total_reward = 0

print("Path : ")

while not done and step < MAX_EVAL_STEPS:
    best_actions = np.flatnonzero(Q[state] == Q[state].max())
    action = np.random.choice(best_actions)

    next_state, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated

    print(f"Step : {step} | State : {state} | Action : {action_names[action]} -> Next_state : {next_state} | Reward : {reward}")

    step += 1
    state = next_state
    total_reward += reward

eval_env.close()

print(f"Total cumulative reward : {total_reward}")








