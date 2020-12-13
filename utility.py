import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


### return the 42500 image data set here
class MyCifar10(Dataset):
    def __init__(self, path, transform, train=True):
        self.cifar10 = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
        self.transforms = transform
        self.least_image = 2500
        self.newset = self.create_new_set()
    def create_new_set(self):
        bird = []
        deer = []
        truck = []
        other = []
        for t in self.cifar10:
            if t[1] == 2:
                bird.append(t)
            elif t[1] == 4:
                deer.append(t)
            elif t[1] == 9:
                truck.append(t)
            else:
                other.append(t)
        random.seed(10)
        bird_ = random.sample(bird, self.least_image)
        random.seed(20)
        deer_ = random.sample(deer, self.least_image)
        random.seed(30)
        truck_ = random.sample(truck, self.least_image)
        data = other + bird_ + deer_ + truck_
        random.seed(40)
        data = random.sample(data, len(data))
        return data
    def __len__(self):
        return len(self.newset)
    def __getitem__(self, index):
        im, label = self.newset[index]
        return self.transforms(im), label

### return the training set and testing set with calculating mean and std
class Cifar:
    def __init__(self, batch_size, threads, less_data, c, f, e):
        mean, std = self._get_statistics(less_data)
        T = []
        O = [transforms.ToTensor(),transforms.Normalize(mean, std)]
        C = [transforms.RandomCrop(size=(32, 32), padding=4)]
        F = [transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.05)]
        E = [transforms.RandomErasing(p=0.5, scale=(0.02,0.4), ratio=(0.3, 1/0.3), value='random')]
        if c:
            T += C
        if f:
            T += F
        T += O
        if e:
            T += E
        train_transform = transforms.Compose(T)
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        #train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        #train_set = MyCifar10(path='./data', transform=train_transform)
        #test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        if less_data:
            self.train = MyCifar10(path='./data', transform=train_transform)
        else:
            self.train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        #self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        #self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self, less_data):
        if less_data:
            train_set = MyCifar10(path='./data', transform=transforms.ToTensor())
        else:
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        #train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())
        test_set = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)] + [d[0] for d in DataLoader(test_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

### function for random oversampling
def weight_sampler(trainset):
    class_count = list(0 for i in range(10))
    target = []
    for im, label in trainset:
        class_count[label] += 1
        target.append(label)
    targets = np.array(target)
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler