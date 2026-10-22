# ---- physical parameters ---------------------------------------------------
L, TFIN, NU = 1.0 , .1 , 0.01

# ---- discretization (used in the PDE residual loss and collocation sampling)
HX, HT = L / 512, 1.0e-3

# ---- collocation sampling --------------------------------------------------
N_COLLOCATION = 100

# ---- reproducibility --------------------------------------------------------
SEED = 42

# ---- training -------------------------------------------------------------
EPOCHS, LR = 10000, 1.0e-2


# ---- architecture and training -------------------plot_solution-----------------------
HIDDEN_NEURONS, HIDDEN_DEPTH = 50, 1

PRE_NEURONS, PRE_DEPTH = HIDDEN_NEURONS,HIDDEN_DEPTH
POST_NEURONS, POST_DEPTH =  HIDDEN_NEURONS,HIDDEN_DEPTH


SAVE_EXTRAS = True  # set True to also save the model checkpoint and loss plot

