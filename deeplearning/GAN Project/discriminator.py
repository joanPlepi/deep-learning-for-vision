import torch 
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self, ngpu):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu

	# Splitted in different layers because we have to extract the features. 
		self.conv_1 = nn.Sequential( 
			# input is (nc) x 64 x 64
	    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
	    nn.LeakyReLU(0.2, inplace=True)
	    )
		
		self.conv_2 = nn.Sequential(
			 # state size. (ndf) x 32 x 32
	    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(ndf * 2),
	    nn.LeakyReLU(0.2, inplace=True)
			)
		
		self.conv_3 = nn.Sequential(
			# state size. (ndf*2) x 16 x 16
	    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(ndf * 4),
	    nn.LeakyReLU(0.2, inplace=True)
	    )

		self.conv_4 = nn.Sequential(
			# state size. (ndf*4) x 8 x 8
	    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
	    nn.BatchNorm2d(ndf * 8),
	    nn.LeakyReLU(0.2, inplace=True)
			)

		self.conv_5_sigmoid = nn.Sequential(
			# state size. (ndf*8) x 4 x 4
	    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
	    nn.Sigmoid()
			)

        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )

	def forward(self, input):

		output = self.conv_1(input)
		output = self.conv_2(output)
		output = self.conv_3(output)
		output = self.conv_4(output)
		output = self.conv_5_sigmoid(output)

		# if input.is_cuda and self.ngpu > 1:
		#     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		# else:
		#     output = self.main(input)

		return output.view(-1, 1).squeeze(1)
		
	def extract_features(self, input):
        out_1 = self.conv_1(input)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out_4 = self.conv_4(out_3)

        max_pool_1 = nn.MaxPool2d(int(out_1.size(2) / 4))
        max_pool_2 = nn.MaxPool2d(int(out_2.size(2) / 4))
        max_pool_3 = nn.MaxPool2d(int(out_3.size(2) / 4))
        max_pool_4 = nn.MaxPool2d(int(out_4.size(2) / 4))
        
        out_1 = max_pool_1(out_1)
        out_2 = max_pool_2(out_2)
        out_3 = max_pool_3(out_3)
        out_4 = max_pool_4(out_4)
        #print(out_1.size(), out_2.size(), out_3.size(), out_4.size())

        
        out_1 = out_1.view(out_1.size(0), -1).squeeze(1)
        out_2 = out_2.view(out_2.size(0), -1).squeeze(1)
        out_3 = out_3.view(out_3.size(0), -1).squeeze(1)
        out_4 = out_4.view(out_4.size(0), -1).squeeze(1)
       # print(out_1.size(), out_2.size(), out_3.size(), out_4.size())

        output = torch.cat((out_1, out_2, out_3, out_4), 1)
        #print(output.size())
        return output
