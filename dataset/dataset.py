import numpy as np
import torch
from torch.utils.data import Dataset

SEQ = {
    'TRAIN': ['00', '02', '05', '06', '07', '08'],
    'VALID': ['01', '03', '04'],
    'TEST' : ['09', '10'],
    'REST': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
}

class SemData(Dataset):
    def __init__(self, split='TRAIN', shuffle=True):
        self.split = split
        self.shuffle = shuffle
        self.data = []
        self.make_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index][:, :, :3]
        image = np.moveaxis(image, -1, 0)
        label = self.data[index][:, :, -1]
        if self.split == 'REST':
            label = np.zeros((33, 513))
        return torch.from_numpy(image).float(), torch.from_numpy(label)

    def make_dataset(self):
        self.data = None

        for seq in SEQ[self.split]:
            path = f'./dataset/simplified/{seq}'
            if self.split == 'REST':
                xyz, extra = np.load(f'{path}/xyz.npy'), np.load(f'{path}/range.npy')
            else:
                xyz, extra, labels = np.load(f'{path}/xyz.npy'), np.load(f'{path}/range.npy'), np.load(f'{path}/labels.npy')
            
            xyz = xyz[:, :, :, 1:]
            extra = np.expand_dims(extra, axis=3)

            if self.split == 'REST':
                concated = np.concatenate([xyz, extra], axis=3)
            else:
                labels = np.expand_dims(labels, axis=3)
                concated = np.concatenate([xyz, extra, labels], axis=3)

            if self.data is None:
                self.data = concated
            else:
                self.data = np.append(self.data, concated, axis=0)
        
        if self.shuffle:
            np.random.shuffle(self.data)
        print(f"Loaded {self.data.shape[0]} samples")
