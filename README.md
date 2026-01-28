# Reiforcement Learning using Tabular Q-Learning on Frozen Lake

This project allowed me to learn the fundamentals of reiforcement learning, by implementing tabular Q-Learning on Frozen Lake(link of the environment below ), using gymnasium library.  

The environnment is a 4x4 frozen lake grid, the agent starts at position [0, 0] and the goal is at the position [4,4]. Some tiles of the grid are "wholes".     
In a first part, to achieve the goal, the agent must step tile by tile while avoiding wholes until he reaches the goal tile.    
In a second part, the lake becomes slippery (argument **is_slippery = True** ) so the agent may moves perpendicular to the intented direction sometimes, and therefore has to take into account these new conditions in order to still reach the goal.      
The 2 parts allowed me to illustrate and understand the differences between a deterministic and a stochastic environment.   

## Running the project 



