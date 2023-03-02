# Descriptor-Conditioned Gradients MAP-Elites

Repository for [MAP-Elites with Descriptor-Conditioned Gradients and Archive Distillation into a Single Policy](https://arxiv.org/abs/) paper, introducing the _Descriptor-Conditioned Gradients MAP-Elites_ algorithm (DCG-MAP-Elites). This builds on top of the [QDax](https://github.com/adaptive-intelligent-robotics/QDax) framework and includes four baselines and three ablation studies:

- [MAP-Elites](https://arxiv.org/abs/1610.05729)
- [MAP-Elites ES](https://arxiv.org/abs/2003.01825)
- [PGA-MAP-Elites](https://hal.archives-ouvertes.fr/hal-03135723v2/file/PGA_MAP_Elites_GECCO.pdf)
- [QD-PG](https://arxiv.org/abs/2006.08505)
- Ablation 1: DCG-MAP-Elites without actor evaluation and without negative samples
- Ablation 2: DCG-MAP-Elites without actor evaluation but with negative samples
- Ablation 3: DCG-MAP-Elites without a descriptor-conditioned actor

DCG-MAP-Elites builds upon PGA-MAP-Elites algorithm and introduces two contributions:

1. The Policy Gradient variation operator is enhanced with a descriptor-conditioned critic that provides gradients depending on a targeted descriptor.
2. Concurrently to the critic's training, the knowledge of the archive is distilled in the descriptor-conditioned actor at no additional cost. This single versatile policy can execute the entire range of behaviors contained in the archive.

<p align="center">
<img width="800" alt="teaser" src="https://user-images.githubusercontent.com/49123210/222401712-fa657210-ce2b-4155-a9cb-5189e281b039.svg">
</p>

## Installation

To run this code, you need to clone the repository and install the required libraries with:
```bash
git clone ...
pip install -r requirements.txt
```

However, we recommend using a containerized environment such as Docker or Singularity. Further details are provided in the last section.

## Usage

To run DCG-MAP-Elites or any other algorithm mentioned in the paper, you just need to run the relevant main script. For example, to run DCG-MAP-Elites, you can run:
```bash
python3 main_dcg_me.py
```

Or to run the MAP-Elites algorithm:
```bash
python3 main_me.py
```

The hyperparameters of the algorithms can be found and modified in the `configs` directory of the repository. Alternatively, they can be modified directly in the command line. For example, to increase the `num_critic_training_steps` parameter to 3000 in DCG-MAP-Elites, you can run:

```bash
python3 main_dcg_me.py num_critic_training_steps=3000
```

Running each algorithm automatically saves metrics, visualisations and plots of performance into the `outputs` directory.

## Singularity

To build a container using Singularity make sure you are in the root of the repository and then run:

```bash
singularity build --fakeroot --force singularity/container.sif singularity/singularity.def
```

You can execute the container with:

```bash
singularity -d run --app [APP NAME] --cleanenv --containall --no-home --nv container.sif [EXTRA ARGUMENTS]
```

where 
- [APP NAME] is the name of the experiment you want to run, as specified by `%apprun` in the `singularity/singularity.def` file. There is a specific `%apprun` for each of the algorithms, ablations and baselines mentioned in the paper.
- [EXTRA ARGUMENTS] is a list of any futher arguments that you want to add. For example, you may want to change the random seed or Brax environment.
