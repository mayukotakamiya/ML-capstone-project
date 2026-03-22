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
   
Round 1 – Geometry-Based Exploration

In the first round, I implemented a distance-based heuristic, which is exploitation as sample near the best observed point and exploration: sample far from existing points. Candidate points were generated within observed bounds and scored using a weighted combination of proximity to the best point and distance from previous samples. This provided structured sampling while maintaining diversity.

Round 2 – Adaptive Weighting

After receiving new outputs, I analysed improvement magnitude (by comparing the old best point vs new best point),  output range and standard deviation. If improvement was large compared to the output range or standard deviation, I increased exploitation weight. If improvement stagnated, I increased exploration weight. This introduced adaptive behaviour rather than fixed heuristics.

My strategy evolves dynamically. In the first query, I used a balanced search (exploitation: exploration = 6:4). In the second query, depending on the improvement on the best point, I adjusted the ratio of exploitation and exploration as explained above.

As the dataset grows, I may incorporate some new approaches such as Gaussian Process regression, Bayesian optimisation acquisition functions (Expected Improvement), and local quadratic approximations. The goal is to move from heuristic search toward probabilistic modelling.
