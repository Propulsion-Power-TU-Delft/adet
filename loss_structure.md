# General updates

## Humid-air mixture PR
- Store N equations of state
- Young regression test
- Eos composition on first node

## Alg EOS
- Group contribution method PCP SAFT
- **Fluid optimization given a impeller**
    - MS thesis?

## Ruben
- Vatas machine
- Standardize discretized components
- Multi-node components

## Compressors
- Add Eckart test case
- Geometric ramping of wake fraction
- LES loss breakdown of the Sandia compressor


# Loss re-organization 

## General points
Fix a formatting norm e.g. in contributing.md

- In the documentation
    - Table of models
    - with **ORIGINAL** sources
    - With the equation of the paper

- Create descriptions of models/move them into the ./docs 
- Leave only the mathematical physical meaning in docs
- Implementation details ONLY in the docstrings



## Equations
- Move hydraulic lengths

## Losses
Split into
`adet.losses.compressors.centrifugal `
`adet.deviation`

#### General

### Stationary
#### Volute
#### Diffuser

### Compressors
#### Shared
#### Centrifugal
#### Axial
#### Mixed

### Turbines
#### Shared
#### Centripetal
#### Axial
#### Mixed

# Testing
- Standardize /data formatting for testcases, JSON schemas
- Write a markdown file for test specs

## Unit testing
- single row with simple loss reproducibility
- mulit-row simple case -> system assembly checks
- fluid equations of state testing

## Regression testing
- Run cases remotely on PRs
- Loss models testing
    - Set of known input and output values 
    - If the model gets modified => Failure

## Pre-commit hooks 
- DCC
