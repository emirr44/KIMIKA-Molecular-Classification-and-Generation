## Model Architecture

Our classification model uses the GraphMVP architecture as the backbone. GraphMVP is a 2D GNN pre-trained with 3D molecular geometry. In practice, we implement a Graph Neural Network (e.g. a Graph Isomorphism Network or GraphConv) that takes each molecule’s graph (atoms + bonds) and outputs a fixed-size embedding vector. During fine tuning on BACE, we freeze this encoder and attach simple MLP head for binary classification. 

## Training and Optimization Details

As previously mentioned, GraphMVP encoder is fine tuned with the BACE training set. The graph data loader constructs PyTorch Geometric Data objects from SMILES (via RDKit). For optimization, we use binary cross-entropy loss on the classifier output. We train with the Adam optimizer (learning rate ~1e-4) for 10 epochs (as results were almost identical compared with training on 5 and 20 epochs), monitoring validation AUC. Dropout is applied in the MLP head for regularization. Class weights are used if needed to balance the 0/1 labels. We experiment with freezing vs. unfreezing the pre-trained layers; typically, Graph Neural Network was freezed to preserve previously learned patterns.

## Evaluation and Conclusion

For simplicity, we measure ROC AUC (Receiver Operating Characteristic Area Under Curve) score on the held-out test set. For comparison, we include baseline models without pretrained weights. 


The fine-tuned GraphMVP substantially outperforms baselines, demonstrating the benefit of pretraining. However, these metrics do not indicate that model works as intended. Modeling graph realtionships is quite difficult task, and because of limitations in data, model still struggles to adapt to given molecules and fully learns patterns.
