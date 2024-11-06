### Note from Carter
Model-free: This approach learns the expected value (Q-value) for each state-action pair without modeling state transition probabilities. The update rule for a given state-action pair, Q(S_t, A_t), is shown below. Here, R_{t+1} represents the reward at the next time step, and max(Q(s_{t+1}, a)) is the maximum Q-value at the future time step. The parameter lambda (λ) represents the discount factor, controlling the weight of future rewards relative to immediate rewards. A higher lambda value places more emphasis on long-term rewards, while a lower value favors short-term gains.

Q(S_t, A_t) = Q(S_t, A_t) + learning_rate * [R_{t+1} + λ * max(Q(s_{t+1}, a_{t+1})) - Q(S_t, A_t)]

At each time step, the probability of each action is calculated using the softmax of the Q-values.

# Q-Learning Table for Advice-Taking Task

## Overview

    This document provides a basic infomaiton of impeement RL model free model for advise task.
---

⚠️ **Important:** Are the State and aciton correct?
⚠️ **Important:** Can define the numerate by myself?

## States

The task has five main states:

1. **Start/Root**: The initial state of the agent.
2. **Win**: The state representing a successful decision that leads to a reward.
3. **Lose**: The state where the agent makes an incorrect decision.
4. **Advise Left**: The state where advice suggests choosing the left option.
5. **Advise Right**: The state where advice suggests choosing the right option.

---

## Actions

Each state offers three possible actions:

1. **Choose Left**: The agent chooses the left option.
2. **Choose Right**: The agent chooses the right option.
3. **Take Advise**: The agent decides to follow the given advice.

---

## Reward Structure
⚠️ **Important:** Is the Reward defined Correct? for advise make it -1?

Rewards are given based on the outcomes of actions in specific states:

- **Win**: +40/+80
- **Take Advise & Win**: +20/+40
- **Lose**: -40 or -80 (penalty)
- **Advise Left** and **Advise Right**: -1 (default cost for taking advice)

### Reward Table

| State        | Start   |      Win     | Lose      | Advise Left | Advise Right |
|--------------|---------|--------------|-----------|-------------|--------------|
| **Reward**   | -1      | +20/+40/+80  | -40/-80   | -1          | -1           |

---

## Parameters

⚠️ **Important:** Should invese tempare be part of parameters?
⚠️ **Important:** If yes, it will only be using after softmax during simpling?

### 1. **Discount Factor (λ)**  
   - Controls the weight of future rewards relative to immediate rewards.
   - A higher λ emphasizes long-term rewards, while a lower λ favors short-term gains.

### 2. **Learning Rates**  
   - Specific learning rates are used for different scenarios:
     - **Win**: Learning rate when a winning action is taken.
     - **Lose**: Learning rate when an action results in loss.
     - **Choose Left with Advise**: Learning rate for choosing left after receiving advice.
     - **Choose Left without Advise**: Learning rate for choosing left without receiving advice.
     - **Choose Right with Advise**: Learning rate for choosing right after receiving advice.
     - **Choose Right without Advise**: Learning rate for choosing right without advice.

---

## State-Action Function \( Q(s, a) \)

The Q-value represents how good it is to be in a state \( S \) and take action \( a \) at that time.

### Bellman Equation

The Q-value is calculated using the Bellman equation:

\[
Q(s_t, a_t) = R_{t+1} + \lambda \cdot \max(Q(s_{t+1}, a_{t+1}))
\]

### Temporal Difference (TD) Error

The error term measures the difference between the expected Q-value and the actual outcome:

\[
\text{error} = R_{t+1} + \lambda \cdot \max(Q(s_{t+1}, a_{t+1})) - Q(s_t, a_t)
\]

### Q-Table Update Rule

The Q-table is updated iteratively based on the TD error:

\[
Q(s_t, a_t) = Q(s_t, a_t) + \text{learning\_rate} \times \left[\text{error}\right]
\]

---

## Action Selection Strategy

A softmax function is applied to the Q-values of the current state to determine the probability distribution over actions. This enables exploration by probabilistically sampling an action based on the Q-values.

---

## Algorithm Steps

1. **Initialize Q-table**: Randomly initialize the Q-values for each state-action pair.
   
2. **Sample Aciton**: Use bellman eqution to get the diffent Q-value for aviliable Action then use softmax to sample one aciton

3. **Calculate TD Error**: Measure the difference between the Q-value computed and the initialized Q-table value.

4. **Update Q-Table**: Apply the TD error to iteratively update the Q-values in the table.

5. **Next State**: Move Agent to next State, if win or lose state, start next trial


⚠️ **Important:** How to simulataion
⚠️ **Important:** Is the Q table random seed need to be fixed?
⚠️ **Important:** Why the SPM is needed during the fit process?