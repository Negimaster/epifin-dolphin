epifin-dolphin
==============

# Authors

- geoffrey.bossut
- guillaume.blassel
- francois.te

# Setup

## Install

Run the following commands (optionally in a venv)

```sh
pip3 install -r requirements.txt
```

## env

create a .env file like so:

```
DOLPHIN_USERNAME=EPITA_GROUPE5
DOLPHIN_PASSWORD=pwd
```

# Run

## Project demo

Run the project the first time to create files with the API data, then run again.

```sh
python3 treecombi.py
Launch ?n
python3 treecombi.py
Launch ?y
Push ?n
```

## Network demo

```sh
python3 network.py
```

# Tests

```sh
pytest tests.py
```

# Remarques

Nous travaillons principalement sur Google Colab
