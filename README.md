# Molecular Graph Representation and Classification 

This repository provides the source code for fine tuning pretrained Graph Neural Network (GraphMVP) for __molecular 
classification__ and utilizes Variational Autoencoder to __generate novel molecules__ from SMILES strings. The goal is to
apply deep learning for drug discovery; we predict β-secretase (BACE1) inhibitor activity (classification) and explore 
new compound structures (generation). The classification task helps identify potential drug candidates, while the
generative model allows de novo design of novel molecules. Screening vast chemical libraries is quite expensive, and
AI offers cost-effective way to explore existing and even propose new molecules, learning patters unseen by humans.

## Attention

This project adapts and uses code from [Shengchao Liu's repository] (https://github.com/chao1224/GraphMVP), which is licensed under the MIT License. List of references can be found in text file in src file

# Problem Definition and Motivation

**Molecular Classification** - We aim to predict whether a given molecule inhibits human β-secretase 1 (BACE1), a key enzyme in Alzheimer’s disease plaque formation. Accurate classification models are important because they can prioritize lead compounds and reduce wet-lab 
costs. The BACE dataset contains 1,513 compounds with binary activity labels (active/inactive). A major challenge is that chemical space is
enormous (∼10^60 possible drug-like molecules), so learning from limited examples is hard. 

**Molecular Generation** - We decided to use a variational autoencoder to map discrete molecules (SMILES strings) into a continuous "chemical" latent space. In this space, we can interpolate and optimize to generate new compounds. Continuous latent representations
make it possible to apply gradient-based search over chemical space. Architecture of variational autoencoder makes it possible
to propose novel molecules not seen in training data that follow molecular features

Molecules are natural graphs (atoms represent nodes, atomic bonds edges) and because of that, Graph Neural Network excel at capturing molecular structure by learning local atom-bond interactions. Specifically, GraphMVP leverages multi-view self-supervision to infuse 3D structural knowledge into a 2D GNN encoder, resulting in a powerful feature extractor for downstream tasks. On the other side, VAEs are superior compared to traditional methods for searching in discrete chemical space, because they learn a continuous embedding of molecules.

# Usage Instructions

To gain more comprehensive understanding of GraphMVP, reader should visit their repository, listed above. There you can find detailed instructions on how to set up environment to work with this model. Nevertheless, we will leave here basic instructions:

### Environment Setup

- Install Anaconda or Miniconda
- Create a new Python 3.8+ environment and install dependencies:
  ```bash
    conda create -n chem_project python=3.8
    conda activate chem_project
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
    conda install -c conda-forge rdkit
    pip install torch-geometric==1.7.2
    pip install -r requirements.txt
    ```
- Clone repository and merge with GraphMVP

# Conclusion

In this project, we created a dual-model pipeline for drug discovery, including a fine-tuned GraphMVP GNN that achieves strong classification accuracy on the BACE inhibitor dataset, leveraging pre-trained 3D-enhanced embeddings and a SMILES-based VAE that is supposed to generate diverse, valid molecules by exploring the learned latent space.

Even though we failed to create accurate models, we set up basis for meaningful models that can possibly lead drug discovery process in the future, for which a lot of time, knowledge and resources is needed.

There are many extensions possible. We could incorporate multi-task learning or additional properties (solubility, toxicity) into GraphMVP fine-tuning. For the VAE, we could experiment with graph-based decoders (like JT-VAE) to improve validity. Conditional generation (e.g. target-specific VAEs) or integrating a property predictor into training could direct the search toward more potent inhibitors.

However, for now, we conclude that we successfully demonstrated how GNNs and VAEs can be used in future drug discovery, but more generally, in progress of Artificial Intelligence
