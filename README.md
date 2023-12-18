# Communication Cost Reduction for Subgraph Counting under Local Differential Privacy via Hash Functions

This repository is the official implementation of [Communication Cost Reduction for Subgraph Counting under Local Differential Privacy via Hash Functions](https://arxiv.org/abs/2312.07055).

## Requirements

To install requirements:

```setup
python3 -m pip install -r requirements.txt
```

## Datasets

The Wikipedia dataset used for the experiments can be downloaded with the following command.
Additionally, all the information concerning the dataset can be found on [this website](https://snap.stanford.edu/data/wiki-Vote.html).

```graphs
wget -P data/ https://snap.stanford.edu/data/wiki-Vote.txt.gz
```

## Running the experiments

The code to run the experiments on triangle counting is included in the [experience_triangles.py](src%2Fexperience_triangles.py) file.

To display all the options, execute:

```options
>>> python src/experience_triangles.py -h
usage: experience_triangles.py [-h] [-o EXP_NAME] [-g {imdb,gplus,wiki}] [-n GRAPH_SIZE] [-a ALGORITHM] [-e PRIVACY_BUDGET] [-s SAMPLE] [-i NB_ITER]

estimate the number of triangles in a graph

options:
  -h, --help            show this help message and exit
  -o EXP_NAME, --exp_name EXP_NAME
                        name of the experiment
  -g {imdb,gplus,wiki}, --graph {imdb,gplus,wiki}
  -n GRAPH_SIZE, --graph_size GRAPH_SIZE
                        size of the graph to extract
  -a ALGORITHM, --algorithm ALGORITHM
                        algorithm to run
  -e PRIVACY_BUDGET, --privacy_budget PRIVACY_BUDGET
                        privacy budget of the algorithm
  -s SAMPLE, --sample SAMPLE
                        sample factor
  -i NB_ITER, --nb_iter NB_ITER
                        number of runs

```

For example, one could try

```experience
python src/experience_triangles.py -o test -g wiki -n 7115 -a grr_css_smooth -e 1 -s 1024 -i 10
```