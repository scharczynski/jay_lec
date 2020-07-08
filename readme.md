I will outline here the steps required to run and use this code:
Any instructions in tickmarks (`` ``) should be run in the terminal as written.

Prerequisites:
     We need to have the fitting package "maxlikespy" downloaded onto your cluster user and installed. 

        1. Choose where you want to install the package.
        2. ``git clone https://github.com/tcnlab/maxlikespy.git``
        3. ``module load python3``
        4. cd into the cloned directory
        5. ``pip3 install --user .``
   
    We also want the package "autograd" for the solver:
        1. "pip3 install --user autograd"

    Now we want to clone the "project" directory I've made
        1. cd where you want to store these files (anywhere as long as not inside maxlikespy)
        2. ``git clone https://github.com/scharczynski/jay_lec.git``

Usage:
    Running the code requires only one job command from inside the project folder:
        
    ``qsub run_all_cells.sh 0 1749 ./jay_lec.py``
     
    This tells the cluster to submit the job script "run_all_cells.sh" and submit the units in the range 0-1749, using the run script "jay_lec.py"

    The run script should create your chosen save directory if it doesn't exist, but creating it by hand isn't a bad idea.