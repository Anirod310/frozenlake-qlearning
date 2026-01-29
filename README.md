# Reiforcement Learning using Tabular Q-Learning on Frozen Lake


![Frozen Lake](https://gymnasium.farama.org/_images/frozen_lake.gif)

This project allowed me to learn the fundamentals of reinforcement learning by implementing tabular Q-learning on the [Frozen Lake Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) using the Gymnasium library.

The environment consists of a 4×4 frozen lake grid. The agent starts at position [0, 0], and the goal is located at position [3, 3]. Some tiles in the grid are holes. The project is separated in 2 parts :

• In the first part, the agent must reach the goal by moving tile by tile while avoiding holes.

• In the second part, the lake becomes slippery (argument **is_slippery = True**), meaning the agent may sometimes move perpendicular to the intended direction. As a result, the agent must take these new conditions into account in order to still reach the goal.

These two parts allowed me to illustrate and understand the differences between a deterministic and a stochastic environment.   

Feel free to take a look at the [report](report.md), where I documented my progress on the project step by step. 

## Running the project 



