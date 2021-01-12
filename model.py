import random

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import os

import loader
from loader import train_data, validate_data, test_data

import matplotlib.pyplot as plt


BATCH_SIZE = 64
EPOCHS = 5
LR = 0.01
GAMMA = 0.3
N = len(loader.classes)

WEIGHT_DIR = './trained_weights'

use_cuda = torch.cuda.is_available()
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
validate_loader = DataLoader(validate_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)

class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
        self.name = 'fcnn'
        self.main = nn.Sequential(
            nn.Conv1d(1, 32, 90, stride = 6),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 31, stride = 6),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 11, stride = 3),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 7, stride = 2),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 5, stride = 2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.AvgPool1d(47)
        )
        self.fc = nn.Linear(512, N)

    def forward(self, x):
        x = self.main(x)
        # print(x.size())
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

model_cnn = ModelCNN()
if use_cuda: model_cnn.cuda()
optimizer = torch.optim.Adam(model_cnn.parameters(), lr = LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)

acc_val_plot =[]
acc_train_plot=[]

def train(model, epoch):
    correct_pred = 0
    model.train()
    print('Learning rate is:', optimizer.param_groups[0]['lr'])
    for i, (data, target) in enumerate(train_loader):
        target = torch.squeeze(target)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = output.data.max(1, keepdim = True)[1]
        correct_pred +=  pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch [{}], iteration [{}], accuracy [{}], loss [{}]'.format(epoch, i+1, correct_pred/len(train_loader.dataset)*100, loss.item()))
    acc_train_plot.append(correct_pred/len(train_loader.dataset)*100)
    test(model, epoch, validate_loader)
    print('Saving trained model after epoch {}'.format(epoch))

def test(model, epoch, dataloader):
    model.eval()
    correct = 0
    for i, (data, target)  in enumerate(dataloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data =  Variable(data, volatile = True)
        target=Variable(target, volatile = True)
        output = model(data)
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc_val_plot.append(correct/len(dataloader.dataset)*100)
    print('Evaluation: Epoch [{}] Accuracy: [{} / {}]'.format(epoch, correct, len(dataloader.dataset)))

def start_training(epochs = EPOCHS, preload_weights = None):
    if preload_weights is not None:
        model_cnn.load_state_dict(torch.load(preload_weights))
    for i in range(epochs):
        scheduler.step()
        train(model_cnn, i)

def predict(model, test_set):
    predictions = []
    model.eval()

    for (id,data) in enumerate(test_set):
        full_data = os.path.join("test",data)
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data =  Variable(full_data, volatile = True)
        output = model(data)
        pred = output.data.max(1, keepdim = True)[1]
        predictions.append(pred)
    return predictions


start_training()
predictions = predict(model_cnn, test_data)
# predictions =  [random.randrange(1, 30, 1) for i in range(len(test_data))]
output_file = open("test_y", "w")
[output_file.write(f'{test_data.file_list[i]}, {predictions[i]}\n') for i in range(len(test_data))]

plt.plot(acc_val_plot, 'r', label='validation set' )
plt.plot(acc_train_plot, 'b', label='train set' )
plt.legend(loc='best')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title('Model Accuracy')
plt.show()