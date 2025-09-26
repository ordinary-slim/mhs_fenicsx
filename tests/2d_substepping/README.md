``` python3
# inspect command line options
python3 test_2d_substepping.py --help
python3 test_2d_chimera_substepping.py --help

# Unsubstepped run
python3 test_2d_substepping.py -r

# Substeppers without advected subdomain
python3 test_2d_substepping.py -ss # Robin
python3 test_2d_substepping.py -sms # Dirichlet

# Substeppers with advected subdomain
python3 test_2d_chimera_substepping.py -ss # Robin
python3 test_2d_chimera_substepping.py -sms # Dirichlet

# Sanity check
python3 -m pytest .

# Run in parallel
mpirun -n 4 python3 test_2d_chimera_substepping.py -ss
```

Inspect `input.yaml` to modify run options e.g.
``` yaml
driver_type: "robin" # Staggered substepper type, `robin` (Robin-Robin) or `dn` (Dirichlet-Neumann)
aitken_relaxation: True # Include Aitken relaxation for Dirichlet-Neumann substepper
predictor :  {
    enabled : True,
}
# Dimensions of advected subdomain (non-dimensional wrt beam radius)
moving_domain_params : {
  adim_front_len : 2,
  adim_side_len : 3,
  adim_bot_len : 2,
  adim_top_len : 2,
}
```
