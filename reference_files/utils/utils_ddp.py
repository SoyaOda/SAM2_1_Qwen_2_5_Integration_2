"""
DDP学習用ヘルパー関数とクラス
train_ddp.pyで使用するユーティリティ機能
"""

import torch
import time

class AverageMeter:
    """平均値を追跡するクラス"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """進捗表示クラス"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def dict_to_cuda(batch):
    """バッチ内のテンソルをCUDAに転送"""
    cuda_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cuda_batch[key] = value.cuda()
        else:
            cuda_batch[key] = value
    return cuda_batch 