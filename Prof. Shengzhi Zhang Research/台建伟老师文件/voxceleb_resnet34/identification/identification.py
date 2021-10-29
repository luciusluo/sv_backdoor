
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import constants as c
import math
import torch.nn.functional as F
from spectrogram import get_spectrum as wav2spec
from tqdm import tqdm
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# ----------------------------------------------------------------------------------------------- #
# %% dataset define and data loading
class IdentificationDataset(Dataset):

    def __init__(self, path, subset, transform=None):
        # 1. Initialize file paths or a list of file names.
        # subset=1: trainset
        # subset=2: valset
        # subset=3: testset

        iden_split_path = './voxceleb1_iden.txt'
        split = pd.read_table(iden_split_path, sep=' ',
                              header=None, names=['subset', 'path', 'label'])

        self.dataset = list(split[split['subset'] == subset]['path'])
        self.label = list(split[split['subset'] == subset]['label'])

        self.path = path
        self.transform = transform

    def __getitem__(self, item):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        track_path = self.dataset[item]
        audio_path = os.path.join(self.path, 'voxceleb1_wav/', track_path)

        label = self.label[item]
        # wav to spectrum
        spec = wav2spec(audio_path)

        if self.transform:
            spec = self.transform(spec)

        return label, spec

    def __len__(self):
        return len(self.dataset)


class normalize_frames(object):
    def __call__(self, m, epsilon=1e-8):
        mu = np.mean(m, 1, keepdims=True)
        std = np.std(m, 1, keepdims=True)
        return (m - mu) / (std + epsilon)
        # return np.array([(v - np.mean(v)) / max(np.std(v), epsilon) for v in m])


class TruncateInput(object):
    # Random select 3s segment from every utterance
    # If the duration from the selecting begin point to the end is less than 3s, padding by repetition
    # time-reversion with the probability of 50%

    def __init__(self, num_frames=300):
        super(TruncateInput, self).__init__()
        self.num_frames = num_frames

    def __call__(self, frames_features):
        upper = frames_features.shape[1]
        start = np.random.randint(0, upper)   # half-open interval
        end = start + self.num_frames
        if end <= upper:
            input_feature = frames_features[:, start:end]
        else:
            input_feature = np.append(
                frames_features[:, start:], frames_features[:, :end - upper], axis=1)

        # time-reversion
        if np.random.random() < 0.5:
            input_feature = np.flip(input_feature, 1)
        return input_feature


class ToTensor(object):
    # Convert spectogram to tensor
    def __call__(self, spec):
        F, T = spec.shape
        # Now specs are of size (freq, time) and 2D but has to be 3D (channel dim)
        spec = spec.reshape(1, F, T)
        spec = spec.astype(np.float32)
        return torch.from_numpy(spec)


# ----------------------------------------------------------------------------------------------- #
transform = transforms.Compose([
    TruncateInput(),
    normalize_frames(),
    ToTensor()
])
transforms_eval = transforms.Compose([
    normalize_frames(),
    ToTensor()
])

trainset = IdentificationDataset(c.VoxCeleb1_Dir, subset=1, transform=transform)
trainset_loader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=c.BATCH_SIZE,
                                              shuffle=True)
valset = IdentificationDataset(c.VoxCeleb1_Dir, subset=2, transform=transforms_eval)
valset_loader = torch.utils.data.DataLoader(dataset=valset,
                                            batch_size=1,
                                            num_workers=c.NUM_WORKERS,
                                            shuffle=False)
testset = IdentificationDataset(c.VoxCeleb1_Dir, subset=3, transform=transforms_eval)
testset_loader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=1,
                                             num_workers=c.NUM_WORKERS,
                                             shuffle=False)

# ----------------------------------------------------------------------------------------------- #
# %% Define network

class VGGMNet(nn.Module):

    def __init__(self, num_classes=2):
        super(VGGMNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96,
                      kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(4, 1)),
            nn.BatchNorm2d(num_features=4096),
            nn.ReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU()
        )
        self.fc8 = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc6(out)
        # global average pooling layer
        _, _, _, width = out.size()
        self.apool6 = nn.AvgPool2d(kernel_size=(1, width))
        out = self.apool6(out)
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)

        # During training, there is no need for SoftMax because SELoss calculates it
        if self.training:
            return out
        else:
            return self.softmax(out)


class VGGVox2(nn.Module):

    def __init__(self, block, layers, num_classes=1211, embedding_size=512):
        super(VGGVox2, self).__init__()
        self.embedding_size = embedding_size
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.maxpool2 = nn.MaxPool2d(kernel_size=(6, 3), stride=(3, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512 * block.expansion,
                      out_channels=512, kernel_size=(9, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.embedding_layer = nn.Linear(512, self.embedding_size)
        self.classifier_layer = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.maxpool2(out)
        # print(out.size())
        out = self.fc(out)
        # global average pooling layer
        _, _, _, width = out.size()
        self.avgpool = nn.AvgPool2d(kernel_size=(1, width))
        out = self.avgpool(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, phase):
        out = self.forward_once(x)
        if phase == 'triplet':
            out = self.embedding_layer(out)
            out = F.normalize(out, p=2, dim=1)
        # # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        # alpha = 10
        # out = out * alpha
        elif phase == 'pretrain':
            out = self.classifier_layer(out)
        return out


net = VGGMNet(c.NUM_CLASSES)
#from torchvision.models.resnet import BasicBlock
#net = VGGVox2(BasicBlock, [3, 4, 6, 3], num_classes=1251)
torch.backends.cudnn.deterministic = True
net = nn.DataParallel(net, device_ids=c.device_ids)  # Multiple GPUs
net.to(c.device)


# ----------------------------------------------------------------------------------------------- #

# %% Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=c.LR_INIT, momentum=c.MOMENTUM,
                      weight_decay=c.WEIGHT_DECAY)
gamma = 10 ** (np.log10(c.LR_LAST / c.LR_INIT) / (c.EPOCH_NUM - 1))
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

# ----------------------------------------------------------------------------------------------- #

# %% Train
def train():
    last_acc = 0.0
    for epoch in range(c.EPOCH_NUM):
        print('epoch:%d/%d' % (epoch + 1, c.EPOCH_NUM))
        lr_scheduler.step()

        # ------------------------------- train ----------------------------------- #
        net.train()  # batchnorm

        for i, (labels, specs) in tqdm(enumerate(trainset_loader)):
            optimizer.zero_grad()
            labels, specs = labels.to(c.device), specs.to(c.device)
            scores = net(specs)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

        # ----------------------------- validation -------------------------------- #
        net.eval()

        val_top5_accuracy = 0
        val_top1_accuracy = 0

        for _, (label, spec) in tqdm(enumerate(valset_loader)):
            label, spec = label.to(c.device), spec.to(c.device)
            probs = net(spec)

            # calculate Top-5 and Top-1 accuracy
            pred_top5 = probs.topk(5)[1].view(5)

            if label in pred_top5:
                val_top5_accuracy += 1
            if label == pred_top5[0]:
                val_top1_accuracy += 1
        val_top5_accuracy /= len(valset_loader)
        val_top1_accuracy /= len(valset_loader)
        print('valset top 1 accuracy: {}'.format(round(val_top1_accuracy, 3)))
        print('valset top 5 accuracy: {}'.format(round(val_top5_accuracy, 3)))
        print('loss: {}'.format(round(loss.item(), 3)))

        # ------------------------------- test ----------------------------------- #
        if val_top1_accuracy < last_acc:

            test_top5_accuracy = 0
            test_top1_accuracy = 0
            with torch.no_grad():
                for _, (label, spec) in tqdm(enumerate(testset_loader)):
                    label, spec = label.to(c.device), spec.to(c.device)
                    probs = net(spec)

                    # calculate Top-5 and Top-1 accuracy
                    pred_top5 = probs.topk(5)[1].view(5)

                    if label in pred_top5:
                        test_top5_accuracy += 1
                    if label == pred_top5[0]:
                        test_top1_accuracy += 1
                test_top5_accuracy /= len(testset_loader)
                test_top1_accuracy /= len(testset_loader)
                print('testset top 1 accuracy: {}'.format(round(test_top1_accuracy, 3)))
                print('testset top 5 accuracy: {}'.format(round(test_top5_accuracy, 3)))
                print('loss: {}'.format(round(loss.item(), 3)))

        last_acc = val_top1_accuracy


if __name__ =='__main__':
    # trainset = IdentificationDataset(c.VoxCeleb1_Dir, subset=1)
    # print(trainset.dataset[60])
    # print(trainset.label[60])
    train()

