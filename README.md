# 🧠 Knowledge-Based Reinforcement Learning for Playing Zork

> A research project combining **Reinforcement Learning (RL)**, **Natural Language Processing (NLP)**, and **Knowledge Graphs (KGs)** to build an intelligent agent capable of reasoning and acting in the text-based game **Zork**.

---

## 📘 Overview

Text-based games like *Zork* present a unique challenge for RL agents: they must **understand language**, **track world states**, and **plan sequences of actions** based solely on textual descriptions.

To address this, we introduce a **Knowledge-Based RL Architecture** where the agent continuously constructs and updates a **Knowledge Graph (KG)** from the game’s text observations.  
This structured graph representation allows the agent to **reason over objects, relations, and locations**, transforming raw text into an interpretable world model.

---

## 🧩 Architecture Summary

Our architecture is composed of five major components:

### 1. 🗣️ T5-OBSRVR — Knowledge Extraction
A fine-tuned **T5 Transformer** serves as a structured knowledge extractor.  
It converts each textual observation into **triplets** ⟨subject, relation, object⟩, forming the building blocks of the KG.

| Model | Input | Output |
|--------|--------|--------|
| Location Extractor | “You are standing in a forest clearing.” | ⟨you, in, forest⟩ |
| Inventory Extractor | “You are carrying a lantern.” | ⟨you, have, lantern⟩ |
| Surroundings Extractor | “There is a tree here.” | ⟨tree, in, forest⟩ |

Each step also adds **temporal links** such as ⟨new_location, came_from, previous_location⟩, enriching spatial relationships.

These three models were fine-tuned on **TextWorld KG** and **JerichoWorld** datasets for precise triplet generation.

---

### 2. 🌐 Grapher — Dynamic Knowledge Graph Construction

The **Grapher module** maintains a coherent, evolving Knowledge Graph that reflects the agent’s understanding of the world.

It:
- Merges new triplets with the existing graph,  
- Removes redundant or outdated nodes,  
- And ensures logical consistency between entities.

The KG thus becomes a **structured state memory** encoding spatial, temporal, and relational information.

<p align="center">
  <img src="docs/kg_update_example.png" alt="KG Update Example" width="500"/>
</p>

Implemented using **Deep Graph Library (DGL)**, the KG is fed to a **Relational Graph Convolutional Network (R-GCN)** for representation learning.

---

### 3. 🧮 R-GCN — Relational Graph Encoding

The **R-GCN** transforms the KG into a dense graph embedding by aggregating information across connected entities.  
This representation captures relational semantics and context, allowing the agent to “understand” what objects are relevant at each state.

---

### 4. 🎯 Action Scorer & Critic — Decision Layer

- **Action Scorer**: Encodes and ranks valid textual actions using the current KG embedding and attention mechanism.  
- **Critic**: Estimates the value of the current state for policy updates.

The agent learns via **Generalized Advantage Estimation (GAE)** and a combined **policy–value loss**.

---

### 5. 🧭 Intrinsic Motivation

To address **reward sparsity**, we introduced **intrinsic rewards** proportional to knowledge gain:
```math
r_intrinsic = 0.1 × T_new + 0.3 × L_new
