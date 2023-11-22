# 多卡使用，单卡不一定有提速效果, 模型训练时间短于数据预处理加载的, 并且数据会均匀加载到每张卡上
import torchnvjpeg
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from multiprocessing.pool import ThreadPool
from torchvision import transforms
from torchvision.transforms import functional_tensor as FT

class ToNorm:
    def __init__(self, mean=0, std=1) -> None:
        self.norm = transforms.Normalize(mean=mean, std=std)

    def __call__(self, x):
        x = x/255.
        return self.norm(x)

# Example Dataset
class ORDataset(Dataset):
    def __init__(self, root, datalist) -> None:
        super().__init__()
        self.data = datalist
        self.root = root
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        imgspath = self.data[idx]['imgspath']
        label = self.data[idx]['label']
        datas = []
        for imp in imgspath:
            with open(imp, 'rb') as fin:
                datas.append(fin.read())
        return imgspath, datas, label

def collate_fn(batch):
    # print(len(batch), len(batch[0]))
    batch = list(zip(*batch))
    return batch

class GpuLoader:
    """
    dataset:只用于读取原始数据, 目前比较简陋
    """
    def __init__(self, loader, trans, devices_id:list=[0]) -> None:
        self.loader = loader
        self.trans = trans
        self.devices_id = devices_id
        self.pool = ThreadPool(len(self.devices_id))
        self.nvjpg = [torchnvjpeg.NvJpeg(i) for i in devices_id]
        self.numthread = len(self.devices_id)

    def splitIdx(self, a, n):
        k, m = divmod(len(a), n)
        return tuple((list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))))
    
    def single_proc(self, datas, decoder):
        # print('decoder', decoder.get_device_id(), datas)
        imgs = []
        for data in datas:
            if decoder is not None:
                img = decoder.decode(data=data, stream_sync=False)
            else:
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
                img = torch.tensor(img)
            imgs.append(img)
        vid = torch.stack(imgs, dim=0).permute(0, 3, 1, 2)
        if self.trans:
            vid = self.trans(vid)
        vid = vid.transpose(0, 1)
        if vid.shape[1]!=16:
            pad = [0]*6
            pad[5] = 16-vid.shape[1]
            vid = F.pad(vid, pad=pad)
        return vid
    
    def multi_proc(self, args):
        datas, decoder = args
        rets = []
        for data in datas:
            if isinstance(data, list):
                data = self.single_proc(data, decoder)
            else:
                if decoder is not None:
                    img = decoder.decode(data=data, stream_sync=False)
                else:
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
                    img = torch.tensor(img)
                data = img.permute(2, 0, 1)
                if self.trans:
                    data = self.trans(data)
            rets.append(data)
        rets = torch.stack(rets, dim=0)
        return rets


    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iter = iter(self.loader)
        return self
    
    def __next__(self):
        try:
            datas = next(self.iter)
            if self.numthread>1:
                imdatas = self.splitIdx(datas[1], self.numthread)
                datas[1] = self.pool.map(self.multi_proc, zip(imdatas, self.nvjpg))
            else:
                datas[1] = self.multi_proc((datas[1], self.nvjpg[0]))
            datas[-1] = torch.Tensor(np.array(datas[-1])).long()
            return datas
        except:
            raise StopIteration

class NewDP(nn.DataParallel):
    def __init__(self, module:nn.Module, device_ids=None, output_device=None, dim=0) -> None:
        super().__init__(module, device_ids, output_device, dim)

    def forward(self, inputs, **kwargs):
        if not self.device_ids or len(self.device_ids)==0:
            return self.module(inputs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, None)
        return self.gather(outputs, self.output_device)
        


if __name__ == '__main__':
    imgsize = (360, 640)
    trans = transforms.Compose([
        transforms.CenterCrop(imgsize),
        ToNorm(),
    ])

    trainset = ORDataset(root, trainlist)
    testset = ORDataset(root, testlist)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, drop_last=True, 
        collate_fn=collate_fn)
    val_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    # warp with gpu
    devices_id = [i for i in range(torch.cuda.device_count())]
    train_loader = GpuLoader(train_loader, trans, devices_id=devices_id)
    val_loader = GpuLoader(val_loader, trans, devices_id=devices_id)