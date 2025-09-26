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
