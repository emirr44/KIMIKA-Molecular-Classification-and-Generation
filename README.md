# Molecular Graph Representation and Classification 

This repository provides the source code for fine tuning pretrained Graph Neural Network (GraphMVP) for molecular 
classification and utilizes Variational Autoencoder to generate novel molecules from SMILES strings. The goal is to
apply deep learning for drug discovery; we predict β-secretase (BACE1) inhibitor activity (classification) and explore 
new compound structures (generation). The classification task helps identify potential drug candidates, while the
generative model allows de novo design of novel molecules. Screening vast chemical libraries is quite expensive, and
AI offers cost-effective way to explore existing and even propose new molecules, learning patters unseen by humans.

# Attention

This project adapts and uses code from [Shengchao Liu's repository] (https://github.com/chao1224/GraphMVP), which is licensed under the MIT License. Architecture of Graph Neural Networks for prediction and code for loading dataset is used in this project, and such files have been placed in dedicated file. For more information and detailed overview, reader should visit mentioned repository
