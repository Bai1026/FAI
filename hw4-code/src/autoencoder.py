import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        Autoencoder: encode to lower dimension and decode back to original dimension
        """
        super(Autoencoder, self).__init__()

        # encoder: input_dim -> encoding_dim -> encoding_dim//2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )

        # decoder: encoding_dim//2 -> encoding_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        # Hint: a forward pass includes one pass of encoder and decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def fit(self, X, epochs=10, batch_size=32):
        # X: numpy array (not torch.Tensor yet)
        #TODO: 5%
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        # optimizer = optim.Adadelta(self.parameters(), lr=1e-3)
        # optimizer = optim.Adagrad(self.parameters(), lr=1e-3)
        # optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        # optimizer = optim.RMSprop(self.parameters(), lr=1e-3)

        dataloader = DataLoader(dataset=TensorDataset(torch.tensor(X, dtype=torch.float)), batch_size=batch_size, shuffle=False)
        # dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)

        loss_history = []
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad() # zero the parameter gradients
                batch_tensor = torch.cat(batch) # convert batch to tensor
                output = self.forward(batch_tensor) # forward pass
                loss = criterion(batch_tensor, output) # calculate loss
                loss.backward() # backpropagation
                optimizer.step() # update weights
                running_loss += loss.item() # accumulate loss
            
            loss_history.append(running_loss/len(dataloader))
            # print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Autoencoder Loss History")
        plt.savefig("AE_loss_history.png")
        plt.clf()


    def transform(self, X):
        #TODO: 2%
        self.eval()
        # Use encoder to transform data
        with torch.no_grad(): # no need to calculate gradients
            # convert to tensor if input is numpy array
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float)
            # encode data
            encoded = self.encoder(X).detach().numpy()
        return encoded
    
    def reconstruct(self, X):
        #TODO: 2%
        self.eval()
        with torch.no_grad(): # no need to calculate gradients
            # transform data
            encoded = self.transform(X)
            # convert the data to tensor
            encoded_tensor = torch.tensor(encoded, dtype=torch.float)
            # decode the data, from tensor to numpy array
            decoded = self.decoder(encoded_tensor).detach().numpy()
        return decoded


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        #TODO: 3%
        # # add random noise to the input data
        # noise = torch.randn_like(x) * self.noise_factor
        # return x + noise

        # add gaussian noise to the input data
        mean = torch.zeros_like(x) # mean = 0 and shape same as x
        std = torch.full_like(x, self.noise_factor) # standard deviation = noise_factor and shape same as x
        noise = torch.normal(mean=mean, std=std) # generate gaussian noise by normal distribution
        return x + noise
    
    def fit(self, X, epochs=10, batch_size=32):
        # TODO: 4%
        self.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        dataloader = DataLoader(torch.tensor(X, dtype=torch.float), batch_size=batch_size, shuffle=True)
        
        loss_history = []
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch in dataloader:
                noisy_batch = self.add_noise(batch) # add noise to the batch
                optimizer.zero_grad()

                output = self.forward(noisy_batch) # forward pass
                loss = criterion(output, batch) # calculate loss, compare output with original batch
                loss.backward() # backpropagation
                optimizer.step() # update weights
                epoch_loss += loss.item() # accumulate loss

            loss_history.append(epoch_loss/len(dataloader))
            # print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader)}")
        
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Denoising Autoencoder Loss History")
        plt.savefig("DAE_loss_history.png")
        plt.clf()
