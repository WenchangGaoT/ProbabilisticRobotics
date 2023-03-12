# Particle Filter

## Author: Wenchang Gao

### Relevant Files

environment.py implements the simulated environment.
filter.py implements Particle Filter algorithm.
./pics/ stores pictures read by the simulation.
Please make sure that these files under same directory.

### Getting the results

run python3 filter.py to get the filter work. Press any button to make the process go on while the map is drawn. Specifically, pressing 's' would save the current picture.

The environment has four modes: whether the observation is noisy and whether an additional state attribute of gps is given. By default these are both set false. One can change the mode by setting arguments "has_gps" and "mode" in the construction fuction in filter.py.


