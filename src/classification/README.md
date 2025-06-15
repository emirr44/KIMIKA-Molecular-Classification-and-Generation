## Model Architecture

Our classification model is based on a Graph Neural Network (GNN) architecture implemented using PyTorch Geometric. The backbone of the model is a message-passing GNN (e.g., GIN or GraphConv) that processes each molecule’s graph structure (atoms as nodes, bonds as edges) and encodes it into a fixed-size vector representation. On top of this, a graph-level prediction head (GNN_graphpred) performs binary classification.

We support the use of pretrained GNN encoders. If a pretrained model is loaded, the encoder can optionally be frozen during fine-tuning to preserve learned features, while the classification head is trained on the BACE dataset.
## Training and Optimization Details

The dataset is loaded from a CSV of SMILES strings and converted into graph representations via RDKit. We perform stratified splitting into training, validation, and test sets to preserve label distributions.

The model is trained using the AdamW optimizer with a learning rate of 1e-3 and a small weight decay (1e-5). To prevent exploding gradients, gradient clipping with a max norm of 2.0 is applied. Mixed precision training (AMP) is optionally used for faster computation on GPUs.

Training is guided by a binary cross-entropy loss on logits, with a learning rate scheduler (ReduceLROnPlateau) monitoring validation AUC. Dropout is applied in the GNN and MLP layers for regularization. Early stopping is used to prevent overfitting, based on stagnation in validation performance.

## Evaluation and Conclusion

We performe model evaluation on a held-out test set using ROC AUC and accuracy as key metrics. A confusion matrix and ROC curve are also plotted to visualize classification performance.

Our GNN-based model achieves decent results, which validates the effectiveness of message passing netowrks (GNNs) in molecular classification. While pretrained weights can improve performance, challenges remain due to the limited size and variability of the BACE dataset. Modeling complex molecular relationships continues to be a non-trivial task, and performance may stagnate without richer representations or data augmentation strategies.
