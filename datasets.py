import torch
from torchvision import transforms, datasets


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


def test():
    gaussian = MNISTGaussianDataset()
    gaussian_data = gaussian[0]
    print(f"Gaussian Padded Data:\n{gaussian_data}")
    zeropad = MNISTZeroDataset()
    zero_data = zeropad[0]
    print(f"Zero Padded Data:\n{zero_data}")


if __name__ == '__main__':
    test()
