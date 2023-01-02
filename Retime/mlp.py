import torch
class MLP(torch.nn.Module):
    
    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)  
        self.linear2 = torch.nn.Linear(num_h, num_h)  
        self.linear3 = torch.nn.Linear(num_h, num_o)
        self.predict = torch.nn.Linear(num_o, 4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        output = self.predict(x)
        return output