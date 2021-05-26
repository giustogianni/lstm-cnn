import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMCNN(nn.Module):
	def __init__(self, seq_len, n_features, n_out, n_hidden=16):
		super(LSTMCNN, self).__init__()
		self.seq_len, self.n_features, self.n_out, self.n_hidden = seq_len, n_features, n_out, n_hidden
		self.lstm = nn.LSTM(
			input_size = self.n_features,
			hidden_size = self.n_hidden,
			num_layers=1,
			batch_first=True 
		)
		self.dropout = nn.Dropout(p=0.2)
		self.batchnorm1 = nn.BatchNorm1d(32)
		self.batchnorm2 = nn.BatchNorm1d(64)
		self.avgpooling = GlobalAvgPooling1d()
		self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5)
		self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
		self.fc1 = nn.Linear(64 + self.n_hidden, n_out)

	def forward(self, x):
		y, (_, _) = self.lstm(x)
		y = self.dropout(y)
		y = y[:,-1] # take last temporal output
		
		z = x.transpose(2,1)

		z = self.conv1(z)
		z = self.batchnorm1(z)
		z = F.relu(z) 

		z = self.conv2(z)
		z = self.batchnorm2(z)
		z = F.relu(z)

		z = self.avgpooling(z)

		out = torch.cat((y,z), dim=1)

		out = out.view(out.size(0), -1) 
		out = self.fc1(out)

		out = F.log_softmax(out) # dim=1

		return out


class GlobalAvgPooling1d(nn.Module):
	def __init__(self):
		super(GlobalAvgPooling1d, self).__init__()

	def forward(self, x):
		return torch.mean(x, 2)

