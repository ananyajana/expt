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

import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)


b1 = 0.5
b2 = 0.999
n_cpu = 8
channels = 1
#img_size =  28
img_size =  2
latent_dim = 100
n_epochs = 200
batch_size = 64
lr = 0.0002
sample_interval = 400
num_classes = 10



#img_shape = (channels, img_size, img_size)
img_shape = (channels, 1, img_size )

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
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        print(img.size())
        img = img.view(img.size(0), *img_shape)
        print(img.size())
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.common_model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
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



    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        print(img_flat.size())
        print(np.prod(img_shape))
        common_representation = self.common_model(img_flat)
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
    '''
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,generator
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)
'''


def gmm_sample(num_samples, mix_coeffs, mean, cov):
    # draws samples from multinomial distributions according to the probability distribution mix_coeff
    z = np.random.multinomial(num_samples, mix_coeffs)
    samples = np.zeros(shape = [num_samples, len(mean[0])])
    i_start = 0
    labels = []
    for i in range(len(mix_coeffs)):
        for k in range(z[i]):
            labels.append(i)
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(
                mean = np.array(mean)[i, :],
                # np.diag here constructs a diagonal array with the rows of the cov matrix
                cov = np.diag(np.array(cov)[i, :]),
                size = z[i])
        #print(cov)
        #print(np.diag(np.array(cov)[i, :]))
        i_start = i_end
        #plt.scatter(samples[:, 0], samples[:, 1])
        #data_sampler[samples]
    #print(labels)
    return samples, labels



sample_size = 100
mix_coeff = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
r = 2
mean = [[-r, 0], [(r/np.sqrt(2)), (r/np.sqrt(2))], [0, r], [(r/np.sqrt(2)), (-r/np.sqrt(2))], [r, 0], [(-r/np.sqrt(2)), (r/np.sqrt(2))], [0, -r], [(-r/np.sqrt(2)), (-r/np.sqrt(2))]]
#cov =  [[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2],[0.2, 0.2], [0.2, 0.2]]
#cov =  [[0.02, 0.02], [0.02, 0.02],[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02],[0.02, 0.02], [0.02, 0.02]]
v = 0.02
cov =  [[v, v], [v, v], [v, v], [v, v], [v, v], [v, v], [v, v], [v, v]]
#print(np.array(cov)[1, :])
#print(np.diag(np.array(cov)[1, :]))
#dataloader, labels = gmm_sample(sample_size, mix_coeff, mean, cov)
#print(samples)
#print(samples[:, 0])
#print(samples[:, 1])
#plt.axis('equal')
#plt.scatter(samples[:, 0], samples[:, 1])



# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    imgs, labels = gmm_sample(batch_size, mix_coeff, mean, cov)
    labels = Tensor(labels).cuda()
    labels = labels.long()
    imgs = Tensor(imgs)
    #print(imgs.size(0))
    # Adversarial ground truths
    valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
    target = Variable(labels, requires_grad=False)
    fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

    # Configure input
    #print(imgs.type)
    real_imgs = Variable(imgs.type(Tensor))

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()

    # Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
    print(z.size())

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    print(gen_imgs.size())
    discriminator_fake_out, _ = discriminator(gen_imgs)
    g_loss = adversarial_loss(discriminator_fake_out, valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()
    print(real_imgs.size())
    discriminator_real_out, class_outputs = discriminator(real_imgs)

    # Measure discriminator's ability to classify real from generated samples
    fake_d_out, _ = discriminator(gen_imgs.detach())
    real_loss = adversarial_loss(discriminator_real_out, valid)
    fake_loss = adversarial_loss(fake_d_out, fake)
    multiclass_loss = criterion(class_outputs, labels)
    
    d_loss = multiclass_loss + (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    '''print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, n_epochs, i, len(imgs), d_loss.item(), g_loss.item())
    )'''
    print(gen_imgs.size())
    gen_imgs1 = (gen_imgs.detach()).view(gen_imgs.size(0), np.prod(img_shape))
    print(gen_imgs1.size())
    #batches_done = epoch * len(imgs) + i
    #if batches_done % sample_interval == 0:
    #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
    plt.gcf().clear()
    plt.axis('equal')
    plt.scatter(gen_imgs1[:, 0], gen_imgs1[:, 1])
    
