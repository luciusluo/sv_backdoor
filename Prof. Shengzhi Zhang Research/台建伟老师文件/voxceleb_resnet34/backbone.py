import torch.nn as nn
import math
import torch
import constants as c
import torch.nn.functional as F
from torchvision import models

class ReLU(nn.Hardtanh):
    
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)
    
    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
               + inplace_str + ')'


class VGGVox2(nn.Module):
    
    def __init__(self, block, layers, num_classes=1211, embedding_size=512, alpha=10):
        super(VGGVox2, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes=num_classes
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
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.avgpool1 = nn.AvgPool2d(kernel_size=(8, 3), stride=1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512 * block.expansion, out_channels=512 * block.expansion, kernel_size=(16, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        
        self.embedding_layer = nn.Linear(512 * block.expansion, self.embedding_size)
        
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(1)
        # self.alpha = alpha
        
        self.classifier_layer = nn.Linear(self.embedding_size, self.num_classes)
    
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
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = self.avgpool1(out)
        ## fc layer kernel_size: (9, 1) or (16, 1)
        out = self.fc(out)

        # Global average pooling layer
        _, _, _, width = out.size()
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, width))
        out = self.avgpool2(out)
        out = out.view(out.size(0), -1)
        out = self.embedding_layer(out)
        return out
    
    def forward(self, x, phase):
        if phase == 'evaluation':
            _padding_width = x[0, 0, 0, -1]
            out = x[:, :, :, :-1-int(_padding_width.item())]
            out = self.forward_once(out)
            out = F.normalize(out, p=2, dim=1)
            
        elif phase == 'triplet':
            out = self.forward_once(x)
            out = F.normalize(out, p=2, dim=1)
            
        elif phase == 'pretrain':
            out = self.forward_once(x)
            ## Multiply by alpha as suggested in https://arxiv.org/pdf/1703.09507.pdf (L2-SoftMax)
            # out = F.normalize(out, p=2, dim=1)
            # out = out * self.alpha
            out = self.classifier_layer(out)
        else: return 'phase wrong!'
        return out


class VGGVox1(nn.Module):
    
    def __init__(self, num_classes=1211, emb_dim=1024):
        super(VGGVox1, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(4, 1)),
            nn.BatchNorm2d(num_features=4096),
            nn.ReLU(inplace=True)
        )
        self.fc7 = nn.Linear(in_features=4096, out_features=self.emb_dim)
        self.fc8 = nn.Linear(in_features=self.emb_dim, out_features=self.num_classes, bias=False)
    
    def forward_once(self, x):
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
        return out
    
    def forward(self, x, phase):
        if phase == 'evaluation':
            out = self.forward_once(x)
        
        elif phase == 'triplet':
            out = self.forward_once(x)
            out = F.normalize(out, p=2, dim=1)
        
        elif phase == 'train':
            out = self.forward_once(x)
            out = self.fc8(out)
        else:
            raise ValueError('phase wrong!')
        return out

#因为ResNet34包含重复的单元，故用ResidualBlock类来简化代码
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=stride, padding=1, bias=False), #要采样的话在这里改变stride
            nn.BatchNorm2d(num_features=outchannel),#批处理正则化
            nn.ReLU(inplace=True),#激活
            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),#采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(num_features=outchannel),
        )
        self.shortcut = shortcut
    
    def forward(self,x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)#计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)#注意激活

#ResNet类
class ResNet34_Vox1(nn.Module):
    def __init__(self, num_classes=1211, emb_dim=512):
        super(ResNet34_Vox1,self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.for_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )#开始的部分
        self.body = self.makelayers([3,4,6,3])#具有重复模块的部分
        '''
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=512 * [3,4,6,3], out_channels=512 * [3,4,6,3], kernel_size=(16, 1)),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        '''
        expansion = 8
        self.fc1 = nn.Linear(in_features=512*expansion, out_features=self.emb_dim)#末尾的部分
        self.fc2 = nn.Linear(in_features=self.emb_dim, out_features=self.num_classes)#末尾的部分

        
    def makelayers(self,blocklist):#注意传入列表而不是解列表
        self.layers = []
        for index,blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64*2**(index-1),64*2**index,1,2,bias=False),
                    nn.BatchNorm2d(64*2**index)
                )#使得输入输出通道数调整为一致
                self.layers.append(ResidualBlock(64*2**(index-1),64*2**index,2,shortcut))#每次变化通道数时进行下采样
            for i in range(0 if index==0 else 1,blocknum):
                self.layers.append(ResidualBlock(64*2**index,64*2**index,1))
        return nn.Sequential(*self.layers)


    def forward(self, x, phase):
        if phase == 'evaluation':
            x = self.for_input(x)
            x = self.body(x)
            # global average pooling layer
            _, _, _, width = x.size()
            self.avgpool = nn.AvgPool2d(kernel_size=(1, width), stride=1)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.fc1(x)
            return x
        
        elif phase == 'train':
            x = self.for_input(x)
            x = self.body(x)
            # global average pooling layer
            _, _, _, width = x.size()
            self.avgpool = nn.AvgPool2d(kernel_size=(1, width), stride=1)
            x = self.avgpool(x)
            x = x.view(x.size(0),-1)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

        else:
            raise ValueError('phase wrong!')
    

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    
    Reference: https://github.com/adambielski/siamese-triplet
    """
    
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
    
    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings.detach(), target)
        
        if embeddings.is_cuda:
            triplets = triplets.to(c.device)
        
        # l2 distance
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        # ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).pow(.5)
        # an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)
        
        # # cosine similarity
        # cos = torch.nn.CosineSimilarity(dim=1)
        # ap_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
        # an_similarity = cos(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
        # losses = F.relu(an_similarity - ap_similarity + self.margin)
        
        return losses.mean(), len(triplets), ap_distances.mean(), an_distances.mean()
    

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a posi7tive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()