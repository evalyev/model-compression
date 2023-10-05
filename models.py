import torch
import torch.nn as nn


class LeNet_Teacher(nn.Module):

    def __init__(self):
        super(LeNet_Teacher, self).__init__()

        self.conv1 = nn.Conv2d(3, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(5 * 5 * 50, 200)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(200, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        #print(output.shape)
        feature = output.view(-1, 50 * 5 * 5)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        
        return output 
    
    
# LeNet architecture
class LeNet_Student(nn.Module):
    def __init__(self):
        super(LeNet_Student, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x