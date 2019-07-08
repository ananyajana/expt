# author: aniruddha maiti, ananya jana

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)


b1 = 0.5
b2 = 0.999
n_cpu = 8
channels = 2
#img_size =  28
data_size =  1
data_vars = 2   # there are two variables(channels) that represent the gaussian data as the data is 2D
latent_dim = 100
n_epochs = 200
batch_size = 64
lr = 0.0002
sample_interval = 400
#num_classes = 10
num_classes = 8 # number of gausssians in the gaussian mixture model


#img_shape = (channels, img_size, img_size)
data_shape = (data_vars, data_size, data_size)
cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False), # why normalization = False
            *block(128, 256),
            #*block(256, 512),
            #*block(512, 1024),
            #nn.Linear(1024, int(np.prod(img_shape))),
            nn.Linear(256, int(np.prod(data_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        #img = self.model(z)
        #img = img.view(img.size(0), *img_shape)
        #return img
        data = self.model(z)
        data = data.view(img.size(0), *data_shape)
        return data


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.common_model = nn.Sequential(
            #nn.Linear(int(np.prod(img_shape)), 512),
            nn.Linear(int(np.prod(data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
 
        )
        self.discr_linear = nn.Linear(256, 1)
        self.discr_sigmoid = nn.Sigmoid()
        
        self.multi_class = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
        )



    #def forward(self, img):
    def forward(self, data):
        #img_flat = img.view(img.size(0), -1)
        data_flat = data.view(data.size(0), -1)
        #common_representation = self.common_model(img_flat)
        common_representation = self.common_model(data_flat)
        d_out, class_out = self.discr_sigmoid(self.discr_linear(common_representation)), self.multi_class(common_representation)

        return  d_out, class_out


# Loss function
adversarial_loss = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    criterion.cuda()

# Configure data loader
'''os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)'''
        
        
## gmm generation data
sample_size = 100
mix_coeff = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
mean = [[-5, 0], [(5/np.sqrt(2)), (5/np.sqrt(2))], [0, 5], [(5/np.sqrt(2)), (-5/np.sqrt(2))], [5, 0], [(-5/np.sqrt(2)), (5/np.sqrt(2))], [0, -5], [(-5/np.sqrt(2)), (-5/np.sqrt(2))]]
v = 0.02
cov =  [[v, v], [v, v], [v, v], [v, v], [v, v], [v, v], [v, v], [v, v]]
dataloader, gmm_labels = gmm_sample(sample_size, mix_coeff, mean, cov)
plt.axis('equal')
plt.scatter(dataloader[:, 0], dataloader[:, 1])

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    #for i, (imgs, labels) in enumerate(dataloader):
    for i, data_point in enumerate(dataloader):
        gmm_labels = gmm_labels.cuda()
        # Adversarial ground truths
        valid = Variable(Tensor(data_point.size(0), 1).fill_(1.0), requires_grad=False)
        target = Variable(labels, requires_grad=False)
        fake = Variable(Tensor(data_point.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_data_point = Variable(data_point.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (data_point.shape[0], latent_dim))))

        # Generate a batch of images
        gen_data_point = generator(z)

        # Loss measures generator's ability to fool the discriminator
        discriminator_fake_out, _ = discriminator(gen_data_point)
        g_loss = adversarial_loss(discriminator_fake_out, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        discriminator_real_out, class_outputs = discriminator(real_data_point)

        # Measure discriminator's ability to classify real from generated samples
        fake_d_out, _ = discriminator(gen_data_point.detach())
        real_loss = adversarial_loss(discriminator_real_out, valid)
        fake_loss = adversarial_loss(fake_d_out, fake)
        multiclass_loss = criterion(class_outputs, labels)
        
        d_loss = multiclass_loss + (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        #if batches_done % sample_interval == 0:
        #    save_image(gen_data_point.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
