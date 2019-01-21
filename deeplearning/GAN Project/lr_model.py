import torch
import torch.nn as nn


class LRModel(nn.Module):
	def __init__(self, input_dim, output_dim, hidden):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden = hidden
		self.seq = torch.nn.Sequential(
		    nn.Linear(input_dim, hidden),
		    nn.ReLU(),
		    nn.Linear(hidden, output_dim)
		                           )
	def forward(self, x):
		return self.seq(x)