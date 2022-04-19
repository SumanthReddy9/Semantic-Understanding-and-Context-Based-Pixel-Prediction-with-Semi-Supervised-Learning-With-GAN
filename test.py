from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from model import _netGenerator

import utils

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.dataset = "folder"
opt.test_image = "test/134.jpg"
opt.workers = 2
opt.batchSize = 64
opt.imageSize = 128
opt.nz = 100
opt.ngf = 64
opt.ndf = 64
opt.niter = 10
opt.lr = 0.0002
opt.beta1 = 0.5
opt.cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt.ngpu = 1
opt.nc = 3
opt.netG = 'model/netGenerator.pth'
opt.netD = ''
opt.outf = '.'
opt.manualSeed = None
opt.nBottleneck = 4000
opt.overlapPred = 4
opt.nef = 64
opt.wtl2 = 0.999
opt.wtlD = 0.001
print(opt)

netG = _netGenerator(opt)
netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
netG.eval()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


image = utils.load_image(opt.test_image, opt.imageSize)
image = transform(image)
image = image.repeat(1, 1, 1, 1)

input_real = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)

input_real = Variable(input_real)


input_real.data.resize_(image.size()).copy_(image)

fake = netG(input_real)
recon_image = input_real.clone()
recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4+opt.imageSize/2)] = fake.data

utils.save_image('val_real_samples.png',image[0])
utils.save_image('val_recon_samples.png',recon_image.data[0])

