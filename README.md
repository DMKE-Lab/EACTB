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


