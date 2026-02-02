# Reinforcement Learning using Tabular Q-Learning on Frozen Lake


![Frozen Lake](https://gymnasium.farama.org/_images/frozen_lake.gif)

This project allowed me to learn the fundamentals of reinforcement learning by implementing tabular Q-learning on the [Frozen Lake Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) using the Gymnasium library.

The environment consists of a 4×4 frozen lake grid. The agent starts at position [0, 0], and the goal is located at position [3, 3]. Some tiles in the grid are holes. The project is separated in 2 parts :

• In the first part, the agent must reach the goal by moving tile by tile while avoiding holes.

• In the second part, the lake becomes slippery (argument **is_slippery = True**), meaning the agent may sometimes move perpendicular to the intended direction. As a result, the agent must take these new conditions into account in order to still reach the goal.

These two parts allowed me to illustrate and understand the differences between a deterministic and a stochastic environment.   

Feel free to take a look at the [report](report.md), where I documented my progress on the project step by step. 

## Getting Started 

Below are the steps you can follow to test and run the project yourself:

- Clone the repository :
    ```bash
    git clone https://github.com/Anirod310frozenlake-qlearning.git
    cd frozen-lake-rl
    ```

- This project relies on the Python libraries listed in the [requirements.txt](requirements.txt) file. Install all dependencies by running the following command in your environment:
    ```bash
    pip install -r requirements.txt
    ```
- For each part of the project, the **config.py** file contains the parameters used for training and evaluating the policy. Feel free to modify them (as I did) and observe how they affect the training and evaluation process.

## Project Structure

The project is organized as follows:
```bash
.
├── part1_deterministic/
│   ├── env.py          # Frozen Lake environment (deterministic)
│   ├── results/        # Store the results  
│   │   ├── metrics.json # Parameters + results of training
│   │   ├── q_table.npy  # Learned policy
│   ├── train.py        # Training loop
│   ├── evaluate.py     # Policy evaluation
│   └── config.py       # Hyperparameters
│
├── part2_stochastic/
│   ├── env.py          # Slippery Frozen Lake environment (stochastic)
│   ├── results/
│   │   ├── metrics.json
│   │   ├── q_table.npy
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
├── report.md           # Detailed step-by-step report
├── requirements.txt
└── README.md
```
## Key Concepts learned 

Through this project, I gained hand-on exeperience with : 

- Markov Decision Processes
- Tabular Q-learning and Bellman updates
- Exploration vs Exploitation policy
- Impact of stochastic transition on learning
- Importance of the different hyperparameters

## Limitation, Possible Improvements & Next Steps

This project was really fun and instructive to understand the first concepts of RL, however I quickly felt the limitations of this algorithm :
- Q-learning does not scale to large or continuous state spaces
- The agent memory only relies on the Q-table
- Performance rapidly decreases in stochastic environment due to randomness

However we can still think of some improvements and extensions for the project, including :
- Using a larger environment like the 8x8 grid
- Implementing SARSA and comparing it to Q-learning 
- Moving to function approximation (DQN) for larger environments

## Contact

If you have any questions, suggestions, or feedback about this project, feel free to reach out:

- GitHub: https://github.com/Anirod310
- Email: bousek.dorian@gmail.com

I’m always open to discussing reinforcement learning, machine learning, or related topics.
