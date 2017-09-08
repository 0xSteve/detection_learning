# Learning Automata (LA) solutions to the link detection problem in Underwater surveillance networks (USNs)

In this folder we have P-model reinforcement learning schemes in various implementations.

#Learning Model Implementations

All simulations are run with the terminal command:

python3 simulation.py



In this folder we have a few implementations of Linear Reward-Penalty model variants in a P-model environment.

## one_sensor_stationary_dlri 
### Discretized Linear Reward-Inaction (DLRI)

This folder contains a solution using a discretized linear reward-inaction automata. It is not as accurate as the linear reward-inaction, however it converges faster than any other model in this project.  This particular learning automata solution not likely to be useful for the detection learning problem.

## one_sensor_stationary_lri 
### Linear Reward-Inaction (LRI)

This folder contains the solution that is most likely to be useful for the applications of detection learning in USNs.

## one_sensor_stationary_lrp
### Linear Reward-Penalty (LRP)

This an ergodic learning automata, and is likely to be the most useful automata to be applied in a nonstaionary random environment.

# Realistic Learning in a Stationary environment

In this folder we have a simulation using reinforcement learning based on the maximum power of a transmitted signal originating from a simulated two-dimensional isotropic source in a reconstruction of the Bedford Basin, Halifax, NS, Canada.

Environmental data used in the reconstruction was provided courtesy of The DRDC Atlantic.

In this example, we show that the maximum power metric is suitably precise for accurate learning of the link-state in an UASN.

# Visualizations

In this folder we have some visualizations using turtle graphics in python.

Visualizations are run with either:

python3 visualize.py
python3 visualize_2.py