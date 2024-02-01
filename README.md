# Decorrelation using Optimal Transport

## Overview

This repository contains code for implementing a decorrelation classifier using Convex Neural Optimal Transport Solvers (Cnots). The decorrelation classifier is a mathematical framework for reducing the correlation between a classifier's predictions and a protected variable. The paper detailing this method can be found https://arxiv.org/abs/2307.05187.

## Installation

### Pip
A step-by-step guide on how to install the project with pip. You can clone decorrelation repository:

```bash
# Example
git clone git@github.com:malteal/ot-decorrelation.git
cd ot-decorrelation
```

and install the requirements 
```bash
# Install requirements
pip install -r requirements.txt

# -- user install for user if you are in a docker env and -e make it editable
pip install . -e --user
```

## Usage
### Create a loader
The first thing you would need to do, is to create a dataloader in `pipeline.py` that loads your classifier' score distribution.
#### Example
```python
# Load multi-class classifier scores from an HDF file and prepare the data for further analysis.
def load_multi_cls(path_to_file:str, score_dims:int, protect_dims:int,
                   protect_cols:list[str]=["mass"], score_cols:list[str]= ["q_score", "t_score", "w_score"], addi_cols:list[str]=["labels"]
                    ):
    # load the h5 file with contain the (p_qcd, p_t, p_vb, mass, sig_bool)
    data = utils.load_hdf(path)

    # get all keys in the file
    key = list(data.keys())[0] if key is None else key

    if len(list(data.keys()))>1:
        raise KeyError("multiple keys in data")

    # init data from h5 file
    # the columns in the file is already organized as mass, labels, q_score, t_score, w_score
    data = data[key][:]

    # the columns in the file is already organized as mass, labels, q_score, t_score, w_score
    data = pd.DataFrame(data, columns=protect_cols+addi_cols+score_cols)

    # softmax the scores
    data[score_cols] = utils.softmax(data[score_cols])
    
    output = {"mass": data[protect_cols].to_numpy(),
              "labels": data[addi_cols].to_numpy(),
              "scores": data[score_cols].to_numpy()}

    return output
```

The function you make to load your data can be added to `ot_decorrelation.py` in of `pl.load_data()`. The next few lines in `ot_decorrelation.py` properly has to be changes as well to match naming, but after this script should be able to run.

### Train the decorrelation 
When the config has been set, it's time to run the model:

```bash
python run/run_decorrelation.py
```
This decorrelation method is able to decorrelate a continuous feature space against protected attributes with optimal transport. It performs well in the context of jet classification in high energy physics, where classifier scores are desired to be decorrelated from the mass of a jet.

### Configation files
The config file used `ot_decorrelation.py` for training is `/configs/ot_config.yaml`

The configuration files in this project are managed by Hydra-core, a powerful configuration management library. Hydra allows you to organize and override settings in a hierarchical and flexible manner. Let’s break down the key components of the Hydra configuration file

```yaml
hydra:
  run:
    dir: ${save_path}
```
* `hydra`: The root section for Hydra configuration.
* `run`: Configuration for the run mode.
* `dir`: The working directory for the run. It is set to ${save_path}, which is a variable that will be defined later in the configuration.

These are global hydra config settings

Additional important variables in the `/configs/ot_config.yaml`:
```yaml
cvx_dim: 3 # dimension of the scores (in the paper its p_qcd,p_t, p_vb = 3)
noncvx_dim: 1 # dimension of the protected variable (in the paper its mass = 1)

target_distribution: source # base or source have to add uniform/normal/dirichlet/base_normal

train_args: ... # hyperparameter for the training setup

model_args: ... # hyperparameter for the two PICNNs
```
## Contact

If you have any questions, feedback, or issues, please feel free to reach out:

- **Email**: [malte.algren@unige.ch](malte.algren@unige.ch)



