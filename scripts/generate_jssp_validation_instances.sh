#! /bin/sh

# Deterministic instances.
python3 -m jssp.generate_validation_instances\
    --seed 0\
    --n_validation_env 100\
    --duration_type deterministic

# Stochastic instances.
python3 -m jssp.generate_validation_instances\
    --seed 0\
    --n_validation_env 100\
    --generate_duration_bounds 0.05 0.1\
    --duration_type stochastic
