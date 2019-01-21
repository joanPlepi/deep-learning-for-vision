from discriminator import Discriminator
from generator import Generator
import torch

device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")

def getGeneratorModel(netGPath='', ngpu=1):
	netG = Generator(ngpu).to(device)
	netG.apply(weights_init)
	if netGPath != '':
	    netG.load_state_dict(torch.load(netGPath))
	print(netG)
	return netG

def getDiscriminatorModel(netDPath='', ngpu=1):
	netD = Discriminator(ngpu).to(device)
	netD.apply(weights_init)
	if netDPath != '':
	    netD.load_state_dict(torch.load(netDPath))
	print(netD)
	return netD

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)