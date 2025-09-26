``` python3
# Inspect command line options
python3 main.py --help

# Run models
python3 main.py -r # unsubstepped
# Substeppers without advected subdomain
python3 main.py -ss # Robin
python3 main.py -sms # Dirichlet
# Substeppers with advected subdomain
python3 main.py -css # Robin
python3 main.py -csms # Dirichlet
```

Inspect `input.yaml` to modify run options e.g.
``` yaml
num_layers : 20

# Control number of time-steps in terms of cooling and macro-steps
max_timesteps : -1 # -1 is all the simulation, 10 corresponds to a few hatches

substepping_parameters : {
    writepos : True # output post at every substep
}
```
