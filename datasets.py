import torch
from torchvision import transforms, datasets
import sklearn.datasets
import numpy as np


class MNISTGaussianDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        if test:
            self.base_dataset = datasets.MNIST('../input_data', train=False, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            self.base_dataset = datasets.MNIST('../input_data', train=True, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        noise = torch.randn(item[0].shape)
        noise /= 2
        noise += 0.5
        noise = torch.clamp(noise, 0, 1)
        padded = torch.cat((item[0], noise), dim=0)
        item = (padded, item[1])
        return item


class MNISTZeroDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        if test:
            self.base_dataset = datasets.MNIST('../input_data', train=False, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            self.base_dataset = datasets.MNIST('../input_data', train=True, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        zero_padding = torch.zeros_like(item[0])
        padded = torch.cat((item[0], zero_padding), dim=0)
        item = (padded, item[1])
        return item


class CIFAR10ZeroDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        if test:
            self.base_dataset = datasets.CIFAR10('../input_data', train=False, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
        else:
            self.base_dataset = datasets.CIFAR10('../input_data', train=True, download=True,
                                               transform=transforms.Compose([
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()]))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        zero_padding = torch.zeros_like(item[0])
        padded = torch.cat((item[0], zero_padding), dim=0)
        item = (padded, item[1])
        return item


class CIFAR10GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, test=False):
        if test:
            self.base_dataset = datasets.CIFAR10('../input_data', train=False, download=True,
                                               transform=transforms.Compose([
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))
        else:
            self.base_dataset = datasets.CIFAR10('../input_data', train=True, download=True,
                                               transform=transforms.Compose([
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor()]))
                                                    # transforms.Normalize((0.1307,), (0.3081,))]))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        noise = torch.randn(item[0].shape)
        noise /= 2
        noise += 0.5
        noise = torch.clamp(noise, 0, 1)
        padded = torch.cat((item[0], noise), dim=0)
        item = (padded, item[1])
        return item

class TwoMoonsPaddedDataset(torch.utils.data.IterableDataset):
    def __init__(self, size, test=False, total_dimension=2):
        self.size = size
        self.test = test
        self.total_dimension = total_dimension
        self.padding_dimension = self.total_dimension - 2

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = sklearn.datasets.make_moons(n_samples=1, noise=0.1)[0]
        padding = np.zeros(self.padding_dimension)
        return np.concatenate([data, padding])


def test():
    tm1 = TwoMoonsPaddedDataset(100)
    tm_samp = tm1[0]
    print(f"Two moons data:\n{tm_samp}")
    tm2 = TwoMoonsPaddedDataset(100, total_dimension=10)
    tm_samp = tm2[0]
    print(f"Two moons padded data:\n{tm_samp}")
    gaussian = MNISTGaussianDataset()
    gaussian_data = gaussian[0]
    print(f"Gaussian Padded Data:\n{gaussian_data}")
    zeropad = MNISTZeroDataset()
    zero_data = zeropad[0]
    print(f"Zero Padded Data:\n{zero_data}")



if __name__ == '__main__':
    test()
