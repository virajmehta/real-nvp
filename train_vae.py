"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision import datasets
import util
from datasets import MNISTZeroDataset, MNISTGaussianDataset, CIFAR10ZeroDataset, CIFAR10GaussianDataset

from models import MLP_ACVAE, VAELoss
from tqdm import tqdm, trange
from torch.autograd.functional import jacobian


def get_datasets(args):
    if args.dataset == 'mnist':
        if args.padding_type == 'none':
            in_channels = 1
            trainset = datasets.MNIST('../input_data', train=True, download=True,
                                      transform=transforms.Compose([
                                            transforms.ToTensor()]))
            testset = datasets.MNIST('../input_data', train=False, download=True,
                                     transform=transforms.Compose([
                                            transforms.ToTensor()]))
        elif args.padding_type == 'gaussian':
            in_channels = 2
            trainset = MNISTGaussianDataset()
            testset = MNISTGaussianDataset(test=True)
        elif args.padding_type == 'zero':
            in_channels = 2
            trainset = MNISTZeroDataset()
            testset = MNISTZeroDataset(test=True)
    elif args.dataset == 'cifar10':
        if args.padding_type == 'none':
            in_channels = 3
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

            trainset = torchvision.datasets.CIFAR10(root='../input_data', train=True, download=True, transform=transform_train)

            testset = torchvision.datasets.CIFAR10(root='../input_data', train=False, download=True, transform=transform_test)
        elif args.padding_type == 'zero':
            in_channels = 6
            trainset = CIFAR10ZeroDataset()
            testset = CIFAR10ZeroDataset(test=True)
        elif args.padding_type == 'gaussian':
            trainset = CIFAR10GaussianDataset()
            in_channels = 6
            testset = CIFAR10GaussianDataset(test=True)

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=args.num_workers)
    return trainloader, testloader, in_channels


def main(args):
    base_path = util.make_directory(args.name, args.ow)
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    trainloader, testloader, in_channels = get_datasets(args)
    # Model
    print('Building model..')
    net = MLP_ACVAE(num_scales=args.num_scales, in_channels=in_channels, mid_channels=64, num_blocks=args.num_blocks)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark
        pass

    if args.resume:
        # Load checkpoint.
        ckpt_path = base_path / 'ckpts'
        best_path_ckpt = ckpt_path / 'best.pth.tar'
        print(f'Resuming from checkpoint at {best_path_ckpt}')
        checkpoint = torch.load(best_path_ckpt)
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = VAELoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args.num_samples, in_channels, base_path)


def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = net(x, reverse=False)
            loss = loss_fn(x, x_hat, mu, logvar)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))


def sample(net, batch_size, device, in_channels):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, in_channels, 28, 28), dtype=torch.float32, device=device)
    x, _ = net.decode(z)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples, in_channels, base_path):
    global best_loss
    global mean_conds
    net.eval()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            with torch.no_grad():
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    conds = []
    for i in trange(x.shape[0]):
        jac = jacobian(net, x[i:i+1, ...])[0]
        side = jac.shape[2]
        channels = jac.shape[1]
        jac = jac.reshape((channels * side * side, channels * side * side))
        cond = np.linalg.cond(jac.cpu().numpy())
        conds.append(cond)
    mean_conds.append(np.mean(conds))
    print(f"Mean of Condition Numbers: {mean_conds[-1]}")

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        ckpt_path = base_path / 'ckpts'
        ckpt_path.mkdir(exist_ok=True)
        best_path_ckpt = ckpt_path / 'best.pth.tar'
        torch.save(state, best_path_ckpt)
        best_loss = loss_meter.avg

    # Save samples and data
    images = sample(net, num_samples, device, in_channels)
    if images.shape[1] == 2:
        images = images[:, :1, :, :]
    if images.shape[1] == 6:
        images = images[:, :3, :, :]
    samples_path = base_path / 'samples'
    samples_path.mkdir(exist_ok=True)
    epoch_path = samples_path / f'epoch_{epoch}.png'
    conds_path = base_path / 'mean_conds.npy'
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, epoch_path)
    np.save(conds_path, np.array(mean_conds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on MNIST')
    parser.add_argument('name', help="The name of the experiment. Results go in data/{name}")
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_blocks', default=8, type=int, help='Number of residual blocks in s or t nets')
    parser.add_argument('--num_scales', default=2, type=int, help='Number of scale / squeeze transformations in this')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')
    parser.add_argument('--padding_type', default='none', choices=('none', 'zero', 'gaussian'))
    parser.add_argument('-ow', action='store_true', help="Overwrite data in directory")

    best_loss = 0
    mean_conds = []

    main(parser.parse_args())
