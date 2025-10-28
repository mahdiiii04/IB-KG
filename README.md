# 🧠 Knowledge-Based Reinforcement Learning for Playing Zork

## Project Overview:
Text based games represent a testbed for language understanding and decision making in complex scenarios, Zork is one of the most challenging games in that case due to its difficulty even for human players.
This project aims to design an AI model capable of understanding the game's text, convert it into a meaningful representation (Knowledge Graphs) and decide which action to take based on that representation.
Inspired by papers such as KG-A2C, this work aims to design a solution from scratch.

## Keywords:
Artificial Intelligence, Natural Language Processing, Knowledge Extraction, Knowledge Graphs (KGs), Reinforcement Learning.

## Architecture Overview:
The general adopted idea is to using environment’s observation to build and update a knowledge graph that we embed and pass to the action decoder to select the most appropriate action as seen in the figure:

1.Knowledge Extraction (T5-OBSRVR):
This is a custom module made to extract from the game's text information about location, inventory and surroundings; example:
- Location : <you, in, forest>
- Inventory : <you, have, sword>, <you, have torch>
- Surroundings : <tree, in, forest>
This module uses three fine-tuned T5 transformers in order to extract each information seperatly and then combines them into one set of triplets in each step.

2.Knowledge Graph Construction:
The previous extracted triplets are used to construct a knowledge graph (KG) that is updated with each step; example:

3.Relational Graph Convolutional Network (R-GCN):
R-GCN is used to convert the KG into meaningful representation that can be used for learning and inference.

4.Learning Part(RL):
In order to choose the best action for each state, A2C + GAE are used to update the policy and get the optimal decisions.

## Used technologies:
Pytorch, NetworkX, DGL, jericho

## Results:
The model achieved a near SOTA performance with stable learning curve, This result is particularly interesting because the agent adopts a strategy that combines two separate routes to maximize reward in the first part of the game. With lower exploration, the agent could have converged to a simpler strategy: either obtaining only +5 by collecting the egg, or directly opening the window to enter the house and continuing on that path for a reward of +25. Instead, the agent intentionally deviates from the main route to collect the smaller reward before returning to the primary path, demonstrating a sophisticated understanding of the environment’s reward structure.

## Detailed Report:
For more implementation detial, feel free to read the report:

