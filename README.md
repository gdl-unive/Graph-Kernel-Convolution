# Graph Kernel Convolution

# Docker

## Requirements
The repository is made to run in a Docker container, please compile the Dockerfile.

There is also a Docker compose file you can run. At the moment it links to an
image from Github but it can be replaced with a link to the Dockerfile. We
recommend this solution to run the code.

The docker image uses Conda, so if you want to run the code straight from your
machine you can get the various commands to execute from the Dockerfile.

# Stand alone

```bash
conda create -n gkc python=3.8
conda activate gkc
pip3 install -r requirements.txt
pip3 install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-geometric==1.7.0 torch_scatter==2.0.8 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1+cpu.html
pip3 install -r requirements2.txt
```

# How to run
Put the configuration you want to execute in the file `config_0.json`. Then execute in a terminal:

```bash
cd ./src
python3 ./launch_grid.py
```

For the parameters refer to the `train_sbatch_grid.py` file.
