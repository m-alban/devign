Partial Implementation of Devign
===============================

This is a partial implementation of the *Devign* model, a graph neural network
based model that identifies vulnerabilities in functions written in the C
programming language. The paper can be found at `DeepAI 
<https://deepai.org/publication/devign-effective-vulnerability-identification-by-learning-comprehensive-program-semantics-via-graph-neural-networks>`_
or on `arXiv 
<https://arxiv.org/abs/1909.03496>`_.
The `dataset <https://sites.google.com/view/devign>`_.

Installation
------------

Python Packages
###############

From requirements.txt
^^^^^^^^^^^^^^^^^^^^

One option is to install from the requirements.txt file. This can be done with conda with the
following command: 
``conda create --name <env_name> --file requirements.txt -c conda-forge -c pytorch -c rusty1s``

Manually
^^^^^^^^

The following (and dependencies) will be required:

#. `PyTorch <https://pytorch.org/get-started/locally/>`_

#. `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_

#. `Natural Language Toolkit <https://www.nltk.org/>`_

#. Pytorch Lightning

#. Torchmetrics

#. Gensim

#. Numpy

Joern
#####

Joern, is the tool used for generating code property graphs and can be installed according to these 
`instructions <https://docs.joern.io/installation>`_. The data preparation process will check first for
installation in <project root>/joern/, i.e. the Joern executable path will be <project root>/joern/joern/joern-cli/joern.
If Joern has not been installed in <project root>/joern/, then ~/bin/joern/ will be searched in accordance
with the default installation of Joern.

Instructions
------------

Setup
#####

Upon first pulling the project, run ``python main.py setup`` to set up project directories
that will be used for storing unpacked data and embedding models.

Running Processes
#################

Run ``python main.py run -h`` for help. A sample dataset has been split off of the main
dataset to verify that things are set up properly and to assess runtime. Training on this
dataset uses fewer epochs.

Running Joern and training Devign on the full dataset takes a few hours on an RTX 2070 Super.

Preparing The Data
^^^^^^^^^^^^^^^^^^

To prepare the data, unpacking the dataset and running Joern to create the graphs, 
run ``python main.py run sample prepare`` for the sample data or 
``python main.py run full prepare`` for the full dataset. Data preparation can also be
combined with training by running the model with ``--rebuild``.

Running The Model
^^^^^^^^^^^^^^^^^ 
If the corresponding data has been prepared, run
``python main.py run sample model flat`` or ``python main.py run full model flat`` for the baseline model described in the paper, or 
``python main.py run sample model devign`` for the Devign model. Alternatively, you can run
``python main.py run sample model flat --rebuild`` to both prepare the sample data and run the model.