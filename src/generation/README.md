## Model Architecture

The architecture is based on two Long Short-Term Memory (LSTM) networks. The encoder processes the input sequence and compresses it into a latent vector by predicting the mean and log-variance of a Gaussian distribution. The reparameterization trick is used to sample from this distribution in a differentiable way. The decoder is also an LSTM, which receives the latent vector at each time step along with the embedded input tokens, and outputs a probability distribution over the vocabulary at each position.

We modeled out VAE to follow standard chemical VAE framework

## Training and Optimization Details

We trained Variational Autoencoder on augmented version of BACE dataset to learn patterns in drug-like molecules. To prepare the input data, we tokenize each SMILES string using a custom tokenizer that builds a fixed vocabulary from the dataset and includes special tokens: <PAD>, <START>, and <END>. Each molecule is encoded into a fixed-length sequence of token indices for input into the neural network.

The model is trained to minimize a VAE loss consisting of two parts:

- **Reconstruction loss**: Cross-entropy between the predicted token probabilities and the target SMILES sequence
- **KL divergence**: A regularization term encouraging the latent space to follow a standard Gaussian distribution

To stabilize training and encourage better latent representations, we use KL annealing, gradually increasing the weight β of the KL divergence.

Training is performed using the Adam optimizer with a learning rate of 1e-3. To handle exploding gradients often encountered in RNNs, we apply gradient clipping with a max norm of 5. We also implement early stopping based on validation loss to prevent overfitting.

## Evaluation

Implementing VAE was the hardest part of the project because LSTMs, as powerful as they are, still need large dataset and strong hardware foundation to endure training, in order to obtain model that demonstrates strong reconstruction ability. Our results were minimal, often generating 0 valid molecules and collapsing VAE. 

However, this architecture has shown true potential and possible usage of autoencoders, and with enough time and knowledge, this can be strong foundation to more robust model that excels at generating new drug-alike molekules
