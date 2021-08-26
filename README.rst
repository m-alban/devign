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

From requrements.txt
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
Run ``python main.py -h`` for help. The first positional argument is the scope of the data used, either the 
full dataset or a sample. The second positional argument specifies whether you are preparing the data or 
running the model. Thus, you can run ``python main.py sample prepare`` to prepare the sample dataset, and then
run ``python main.py sample model flat`` for the baseline model described in the paper, or 
``python main.py sample model devign`` for the Devign model. Alternatively, you can run
``python main.py sample model flat --rebuild`` to both prepare the sample data and run the model.