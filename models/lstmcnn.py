import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMCNN(nn.Module):
	def __init__(self, seq_len, n_features, n_out, n_hidden=64):
		super(LSTMCNN, self).__init__()
		self.seq_len, self.n_features, self.n_out, self.n_hidden =seq_len, n_features, n_out, n_hidden
		self.lstm = nn.LSTM(
			input_size = self.n_features,
			hidden_size = self.n_hidden,
			num_layers=1,
			batch_first=True 
		)
		self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
		self.fc1 = nn.Linear(..., n_out)

	def forward(self, x):
		y, (_, _) = self.lstm(x)
		
		z = x.transpose(2,1)
		z = self.conv1(x)

		x = torch.cat((y,z), dim=1)
		
		
