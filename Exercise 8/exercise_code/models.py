from operator import ne
import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams

        neurons_hidden_layer = self.hparams["n_hidden"]

        self.encoder = nn.Sequential(
            nn.Linear(input_size, neurons_hidden_layer),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.BatchNorm1d(neurons_hidden_layer),          
            nn.Linear(neurons_hidden_layer, neurons_hidden_layer//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//2),
            nn.Linear(neurons_hidden_layer//2, neurons_hidden_layer//4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//4),
            nn.Linear(neurons_hidden_layer//4, latent_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(latent_dim), 
        )

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.latent_dim = latent_dim 
        self.output_size = output_size
        self.latent_dim = latent_dim    
        neurons_hidden_layer = self.hparams["n_hidden"]

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, neurons_hidden_layer//4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//4),           
            nn.Linear(neurons_hidden_layer//4, neurons_hidden_layer//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//2),
            nn.Linear(neurons_hidden_layer//2, neurons_hidden_layer),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer),
            nn.Linear(neurons_hidden_layer, output_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(output_size), 
        )

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        # feed x into encoder, then decoder!
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)
    
        return reconstruction

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################

        self.train()  # Set the model to training mode
        self.optimizer.zero_grad()  # Reset gradients

        # Extract data from the batch
        images = batch
        images = images.to(self.device)

        # Flatten the images
        images = images.view(images.shape[0], -1)

        # Forward pass
        reconstruction = self(images)

        # Calculate the loss
        loss = loss_func(reconstruction, images)

        # Backward pass
        loss.backward()

        # Update weights
        self.optimizer.step()

        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Extract data from the batch
            images = batch
            images = images.to(self.device)

            # Flatten the images
            images = images.view(images.shape[0], -1)

            # Forward pass
            reconstruction = self(images)

            # Calculate the loss
            loss = loss_func(reconstruction, images)

        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        neurons_hidden_layer = self.hparams["n_hidden"]
        num_classes = self.hparams["num_classes"]
        
        self.model = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, neurons_hidden_layer),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer),  
            nn.Linear(neurons_hidden_layer, neurons_hidden_layer//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//2),
            nn.Linear(neurons_hidden_layer//2, neurons_hidden_layer//4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(neurons_hidden_layer//4),
            nn.Linear(neurons_hidden_layer//4, num_classes),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(num_classes), 
        )

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
