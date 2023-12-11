import torch.nn as nn
import torchvision.models as models

class FFN(nn.Module):
    def __init__(self,class_num,input_dim=52*100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, class_num)
        )

    def forward(self,x):
        x=x.reshape(x.shape[0],-1)
        return self.model(x)

class RNN(nn.Module):
    def __init__(self,class_num,input_dim=52):
        super().__init__()
        self.lstm = nn.RNN(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, class_num)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.fc(x[:,-1,:])
        return y

class LSTM(nn.Module):
    def __init__(self,class_num,input_dim=52):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, class_num)

    def forward(self, x):
        x, _ = self.lstm(x)
        y = self.fc(x[:,-1,:])
        return y

class CNN(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.Linear=nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,class_num)
        )

    def forward(self,x):
        x=x.unsqueeze(1)
        x=self.model(x)
        x=x.reshape(x.shape[0],-1)
        x=x[:,:64]
        x=self.Linear(x)
        return x

class Resnet(nn.Module):
    def __init__(self,class_num,pretrained=True):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, class_num)

    def forward(self,x):
        x=x.unsqueeze(1)
        return self.model(x)

#test
if __name__ == '__main__':
    import torch
    x=torch.rand([2,100,52])
    model=FFN(class_num=2,input_dim=52*100)
    y=model(x)
    print(y.shape)
    model=RNN(class_num=2,input_dim=52)
    y=model(x)
    print(y.shape)
    model=LSTM(class_num=2,input_dim=52)
    y=model(x)
    print(y.shape)
    model=Resnet(class_num=2)
    y=model(x)
    print(y.shape)
    model=CNN(class_num=2)
    y=model(x)
    print(y.shape)

