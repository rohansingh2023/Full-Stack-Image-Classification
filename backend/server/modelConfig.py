import torch.nn as nn
import pathlib
from torchvision.transforms import transforms

train_path = "../data/seg_train/seg_train/"
pred_path = '../data/seg_pred/seg_pred/'

root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

transform = transforms.Compose((
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
))

class NNArch(nn.Module):
    def __init__(self):
        super(NNArch, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (32,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (32,12,150,150)
        self.relu1=nn.ReLU()
        #Shape= (32,12,150,150)
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (32,12,75,75)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (32,20,75,75)
        self.relu2=nn.ReLU()
        #Shape= (32,20,75,75)
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (32,32,75,75)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (32,32,75,75)
        self.relu3=nn.ReLU()
        #Shape= (32,32,75,75)
        self.conv4=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        #Shape= (32,64,75,75)
        self.bn4=nn.BatchNorm2d(num_features=64)
        #Shape= (32,64,75,75)
        self.relu4=nn.ReLU()
        #Shape= (32,64,75,75)
        self.fc=nn.Linear(in_features=75 * 75 * 64,out_features=len(classes))
    def forward(self,x):
        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=self.conv4(output)
        output=self.bn4(output)
        output=self.relu4(output)
        #Above output will be in matrix form, with shape (32,64,75,75)
        output=output.view(-1,64*75*75)
        output=self.fc(output)
        return output