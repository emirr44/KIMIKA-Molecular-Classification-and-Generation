# Molecular Graph Representation and Classification 

This repository provides the source code for fine tuning pretrained Graph Neural Network (GraphMVP) for __molecular 
classification__ and utilizes Variational Autoencoder to __generate novel molecules__ from SMILES strings. The goal is to
apply deep learning for drug discovery; we predict β-secretase (BACE1) inhibitor activity (classification) and explore 
new compound structures (generation). The classification task helps identify potential drug candidates, while the
generative model allows de novo design of novel molecules. Screening vast chemical libraries is quite expensive, and
AI offers cost-effective way to explore existing and even propose new molecules, learning patters unseen by humans.

## Attention

This project adapts and uses code from [Shengchao Liu's repository] (https://github.com/chao1224/GraphMVP), which is licensed under the MIT License. List of references can be found in text file in src file
