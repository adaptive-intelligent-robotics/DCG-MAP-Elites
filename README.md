# Descriptor-Conditioned Gradients MAP-Elites

Repository for:
- [_MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy_](https://dl.acm.org/doi/10.1145/3583131.3590503), introducing **DCG-MAP-Elites GECCO**, and that received a _Best Paper Award_ at GECCO 2023 in Lisbon.
- _Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning_, introducing **DCG-MAP-Elites-AI**, an extension of DCG-MAP-Elites GECCO.

## Summary

DCG-MAP-Elites-AI builds upon PGA-MAP-Elites algorithm and introduces three key contributions:
1. The Policy Gradient variation operator is enhanced with a descriptor-conditioned critic that reconciles diversity search with gradient-based methods coming from reinforcement learning.
2. As a by-product of the critic's training, a descriptor-conditioned actor is trained, at no additional cost, distilling the knowledge of the population into one single versatile policy that can execute a diversity of high-performing behaviors.
3. In turn, we exploit the descriptor-conditioned actor by injecting it in the population, despite network architecture differences.

This repository builds on top of the [QDax](https://github.com/adaptive-intelligent-robotics/QDax) framework and includes four baselines and three ablation studies:

### Baselines

- [MAP-Elites](https://arxiv.org/abs/1504.04909)
- [MAP-Elites ES](https://dl.acm.org/doi/10.1145/3377930.3390217)
- [PGA-MAP-Elites](https://dl.acm.org/doi/10.1145/3449639.3459304)
- [QD-PG](https://dl.acm.org/doi/10.1145/3512290.3528845)

### Ablations

- DCG-MAP-Elites GECCO
- DCG-MAP-Elites without Actor Injection
- DCG-MAP-Elites without a Descriptor-Conditioned Actor

## Installation

To run this code, you need to clone the repository and install the required libraries with:
```bash
git clone https://github.com/adaptive-intelligent-robotics/DCG-MAP-Elites
pip install -r requirements.txt
```

However, we recommend using a containerized environment with Apptainer.

## Apptainer

We provide an Apptainer/Singularity Definition file, to run the source code in a containerized environment in which all the experiments and figures can be reproduced. In the following, make sure you are at the root of the cloned repository.

To build a container using Apptainer/Singularity, use the provided `apptainer/container.def` file:
```bash
apptainer build --fakeroot --force --sandbox apptainer/container.sif apptainer/container.def
```

Then, you can run a shell within the container with:
```bash
apptainer shell --pwd /project/ --bind $(pwd):/project/ --cleanenv --containall --home /tmp/ --no-home --nv --workdir --writable apptainer/ apptainer/container.sif"
```

## Run main experiments

To run any algorithms `<algo>`, on any environments `<env>`:
1. Build a container
2. Run a shell within the container, as explained in the previous section
3. In `/project/`, run `python main.py env=<env> algo=<algo> seed=$RANDOM num_iterations=4000` to run for 1,024,000 evaluations
4. During training, the metrics, visualizations and plots of performance can be found in real time in the `output/` directory

For example, to run DCG-MAP-Elites-AI on Ant Omni:
```
python main.py env=ant_omni algo=dcg_me seed=$RANDOM num_iterations=4000
```

The configurations for all algorithms and all environments can be found in the `configs/` directory. Alternatively, they can be modified directly in the command line. For example, to increase `num_critic_training_steps` to 5000 in PGA-MAP-Elites, you can run:
```bash
python main.py env=walker2d_uni algo=pga_me seed=$RANDOM num_iterations=4000 algo.num_critic_training_steps=5000
```

## Run reproducibility experiments

The reproducibility experiments load the saved archives from the main experiment (see previous section) and evaluate the expected QD score, expected distance to descriptor and expected max fitness of the populations of the different algorithms.

> :warning: Before running a reproducibility experiment, the main experiment for the corresponding environment and algorithm should be completed.

For example, to evaluate the reproducibility for QD-PG on AntTrap Omni, run:
```bash
python main_reproducibility.py env_name=anttrap_omni algo_name=qd_pg
```

The results will be saved in the `output/reproducibility/` directory.

## Figures

Once all the experiments are completed, any figures from the paper can be replicated with the scripts in the `analysis/` directory.

- Figure 1: `analysis/plot_main.py`
- Figure 2: `analysis/plot_archive.py`
- Figure 3: `analysis/plot_ablation.py`
- Figure 4: `analysis/plot_reproducibility.py`
- Figure 5: `analysis/plot_elites.py`

## P-values

Once all the experiments are completed, any p-values from the paper can be replicated with the script `analysis/p_values.py`.
