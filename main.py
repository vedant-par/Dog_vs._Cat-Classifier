import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

input_size = 196608
hidden_size = 64
output_size=2


test_image = Image.open('dog_test.jpg')
test_transforms = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
test_image_tensor = test_transforms(test_image).unsqueeze(0)

data_transforms = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

trainset = datasets.ImageFolder('data/train',transform=data_transforms)
valset = datasets.ImageFolder('data/val',transform=data_transforms)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=96, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=96, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x =  x.view(x.size(0), -1)
        x =  F.relu(self.input(x))
        x =  F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x =  self.fc3(x)
        return F.log_softmax(x, dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 10

def train_model():
    for epoch in range(EPOCHS):
        for data in train_loader:
            X, y = data
            net.zero_grad()
            output = net(X)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        print(loss)
    torch.save(net.state_dict(), 'model.pth')

def accuracy():
    correct=0
    total=0

    with torch.no_grad():
        for data in train_loader:
            X,y = data
            output = net(X.view(-1, 196608))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                total += 1


   

    print("Accuaracy: ",round(correct/total, 3))
#train_model()

def validation():
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for data in val_loader:
            images, labels = data
            output = net(images)
            print(output)
            _, predicted = torch.max(output.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

        print('Validation accuracy: ', round(val_correct / val_total, 3))


net.load_state_dict(torch.load('model.pth'))
# validation()

output= net(test_image_tensor)
_, predicted = torch.max(output.data, 1)
probabilities = F.softmax(output, dim=1)
print('Probabilities: ', probabilities)


print('prediction: ',predicted.item())
print(trainset.classes)