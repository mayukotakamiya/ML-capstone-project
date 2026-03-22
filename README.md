# ML-capstone-project
1. Project Overview
   
This project explores Black-Box Optimisation (BBO) across eight synthetic functions of increasing dimensionality (2D to 8D). Each function mimics real-world optimisation challenges such as non-linearity, noisy outputs and multiple local maxima. The internal structure of each function is unknown, and the only way to gain information is by submitting query points and observing returned outputs.

The overall goal is to maximise each function’s output using a limited number of expensive queries. This reflects real-world scenarios where evaluations are costly (e.g., hyperparameter tuning, experimental design, chemical processes or industrial simulations).

Black-box optimisation is highly relevant in machine learning because many practical problems involve:

Expensive model training cycles
Unknown response surfaces
No analytical gradients
Noisy or stochastic outputs
This capstone strengthens my ability to make data-driven decisions under uncertainty, a skill directly applicable to ML engineering, research and applied optimisation tasks.

2. Inputs and Outputs

Each function receives an input vector, based on the initial input data and output data. High-level summary of each BBO function was provided in advance. Each query returns a single scalar value as an output. The new input data point and new output data point will be appended to the existing input data and output data for a next query. 

3. Challenge Objectives
   
The objective is to maximise the output of each unknown function while not being provided the underlying function or potential noises in outputs. The main challenge lies in balancing exploration (sampling uncertain or underexplored regions) and exploitation (refining around promising regions). 

5. Technical Approach
   
This project follows an iterative black-box optimisation (BBO) strategy, where the approach evolves as more data becomes available. The core principle is to balance exploration (learning the search space) and exploitation (refining promising regions), supported by surrogate modelling techniques.

Queries 1–2: Heuristic Exploration
In the initial stages, I relied on simple heuristics:
Random and space-filling sampling
Distance-based exploration to cover the domain
Light exploitation around high-performing points
This phase prioritised broad exploration, as little was known about the function structure.

Query 3: Structured Search with SVM Concepts
As more data became available, I introduced ideas inspired by Support Vector Machines (SVMs):
Framing the problem as “good vs bad” classification
Identifying boundary regions between high and low outputs
Sampling near these boundaries to improve learning efficiency
This improved the ability to focus on informative regions rather than purely random exploration.

Query 4–5: Neural Network Surrogate Models
From Query 4 onwards, I introduced neural network surrogate models to approximate the unknown functions:
Used MLPRegressor to model nonlinear relationships
Generated candidate points and selected those with highest predicted outputs
Combined local refinement (around best points) with global exploration
Inspired by deep learning concepts:
Feature hierarchies → capturing interactions between inputs
Backpropagation → learning how outputs change with inputs
Architectural trade-offs → balancing model complexity and overfitting

Query 6: CNN-Inspired Trade-offs
In the latest stage, I refined the approach using ideas from convolutional neural networks (CNNs):
Depth vs efficiency
→ Use simple methods for low-dimensional functions (1–2) and more expressive models for higher dimensions (6–8)
Generalisation vs overfitting
→ Maintain exploration to avoid overfitting to known regions
Complexity vs clarity
→ Match model complexity to problem difficulty
This resulted in function-specific strategies:
Low dimensions → simple heuristics
Mid dimensions → small neural networks
High dimensions → more expressive surrogate models with stronger exploration

Across all queries, the optimisation pipeline follows:
1.Load observed data (inputs and outputs)
2.Train surrogate model (if applicable)
3.Generate candidate points -Local (around best point) and Global (across domain)
4.Score candidates using predicted output (exploitation) and distance from known points (exploration)
5.Select next query point
6.Submit and update dataset
