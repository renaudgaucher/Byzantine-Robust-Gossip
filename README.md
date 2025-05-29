# Robust Byzantine Gossip : tight breakdown point and topology aware attacks

Code for ICML2025 paper.

### Software dependencies

Our script derives from https://github.com/LPD-EPFL/robust-collaborative-learning.  The dataset manager part of the script only run on Linux machines. 

We run our script with Python 3.12.3, and the outer librairies are in the config.txt file. 

We list below the OS on which our scripts have been tested:
* CentOS Linux release 7.7.1908

### Hardware dependencies

Although our experiments are time-agnostic, we list below the hardware components used:
* Intel Xeon CPU Gold 6230 20 cores @ 2.1 Ghz with Nvidia Tesla V100 32 GB of RAM
* Intel Xeon CPU Gold 6330 28 cores @ 2.0 Ghz with Nvidia A100 and 32 GB of RAM

  
### Command
All our results can be reproduced using the following.

The 1st and 2nd command reproduce results on MNIST in on the Two Worlds and Erdos Renyi graph, the 3rd and 4th reproduce results on the averaging tasks on Two Worlds and Erdös Renyi graphs (do not requires Linux nor a GPU), and the 5th command reproduces CIFAR-10 results on the Two Worlds graph.

In the root directory:
```
$ python3 reproduce_minst0.py
$ python3 reproduce_minst1.py
$ python3 reproduce_averaging0.py --supercharge 4
$ python3 reproduce_averaging1.py --supercharge 4
$ python3 reproduce_cifar.py
```

Note that experiments on mnist and cifar10 requires non negligible disk space, and requires a linux OS to be launched. 

Depending on the hardware, instructing the script to launch several runs per available GPU may reduce the total runtime.
For instance, to push up to 4 concurrent runs per GPU:
```
$ python3 reproduce.py --supercharge 4
```
