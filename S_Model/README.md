# Learning Automata (LA) solutions to the link detection problem in Underwater surveillance networks (USNs)

This is a continuous control evironment utilizing the S-model of reinforcement learning environment to accurately simulate the continuity of realistic stochastic environments.

# Inputs

As an input this simulation requires a CSV file containing a discrete PDF. Starting from 0m representing the surface of a water column and the end of file representing the maximum depth.

The depth of the water column is inferred from the granularity variable in the simulation, and the number of data points.  For example 70 data points with a granularity of 1 datapoint/m corresponds to a maximum depth of 69m.

# Output files

The outputs are the action probabilities, and learned best depths as a file. These are handled in this way to allow the user to plot in the environment of their choice. In our case, we chose to use MATLAB. In fact, the CSV files are formatted specifically for MATLAB's CSV read function, and may not work off-the-shelf with other applications.