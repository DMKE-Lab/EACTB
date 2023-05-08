## EACTB

Paper: Joint Multi-Feature Information Entity Alignment for Cross-lingual Temporal Knowledge Graph with BERT


This repository contains the implementation of the EACTB architectures described in the paper.

## Installation
* Python 3.x (tested on Python 3.6)

* Tensorflow 2.x 

* Scipy

* Numpy

* argparse

* Pandas

* Scikit-learn

* time

* pickle


```bash
conda create -n openea python=3.6

conda activate openea

conda install tensorflow-gpu==1.12

conda install -c conda-forge graph-tool==2.29

conda install -c conda-forge python-igraph
```



Besides, it is well-recognized to split a dataset into training(20%), validation(10%) and test(70%) sets. 

We use Hits@m (m = 1, 10), and mean reciprocal rank (MRR) as the evaluation metrics.  Higher Hits@m and MRR scores ndicate better performance.

### Train and Test
To run the off-the-shelf approaches on our datasets and reproduce our experiments, change into the ./run/ directory and use the following script:


```bash
python weighted_concat.py -d demo_embd/pairwise_dump.json -g demo_embd/zh_en_graph_embd.pkl -i dbp15k/zh_en/test
```

Use the logic package to export BERT based embeddings. First, install the logic:

```bash
git submodule init

git submodule update
```

In addition, due to the rapid development of logic, it is recommended to submit version d1b5046 instead:

```bash
cd relogic

git checkout d1b5046
```

The parameter "- local_rank" in logic represents the ID of the GPU.
Training BERT requires manual stopping of training

```bash
bash train_ bert.sh 0 zh_ en

bash eval_ bert.sh 0 zh_ en
```

Run the following code on the pychar terminal for testing:

```bash
python weighted_concat.py --desc relogic/saves/pair_matching/zh_en/pairwise_dump.json --graph graph_ckpt/zh_en_graph_embd.pkl --ill data/zh_en/test
```
