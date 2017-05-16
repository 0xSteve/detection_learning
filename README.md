# Learning Automata (LA) solutions to the detection problem in Underwater surveillance networks (USNs)

# one_sensor_stationary_dlri 
## Discretized Linear Reward-Inaction (DLRI)

This folder contains a solution using a discretized linear reward-inaction automata. It is not as accurate as the linear reward-inaction, however it converges faster than any other model in this project.  This particular learning automata solution not likely to be useful for the detection learning problem.

# one_sensor_stationary_lri 
## Linear Reward-Inaction (LRI)

This folder contains the solution that is most likely to be useful for the applications of detection learning in USNs.

# one_sensor_stationary_lrp
## Linear Reward-Penalty (LRP)

This an ergodic learning automata, and is likely to be the most useful automata to be applied in a nonstaionary random environment.

# Running the simulations.

To run a simulation, in the desired simulation folder, open a terminal and type:

python3 simulation.py


# Installation of Dependencies.
All code is written and tested using Python 3.6.x, and requires the following dependencies:
- PIP 3
- scipy for Python 3.6.x (optional)
- numpy for Python 3.6.x 

Installation procedures follow for Debian-like GNU/Linux distributions.

## To install Python 3.6.x run the following command in the terminal:

sudo apt-get install python3

To verify that Python 3.6.x is installed, in the terminal type:

python3 --version

## To install PIP3 run the following command in the terminal:

sudo apt-get install python3-pip

## To install scipy run the following command in the terminal:

sudo pip3 install scipy

## To install numpy run the following command in the terminal:

sudo pip3 install numpy
