from pathlib import Path
from shutil import rmtree


class AverageMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_directory(dirname, overwrite):
    base_path = Path('data')
    this_path = base_path / dirname
    if this_path.exists():
        if overwrite:
            rmtree(this_path)
        else:
            raise ValueError(f"Experiment with name {dirname} exists!")
    this_path.mkdir()
    return this_path
