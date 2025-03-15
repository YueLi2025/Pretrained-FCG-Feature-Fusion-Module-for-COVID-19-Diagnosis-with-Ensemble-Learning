import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def prepare_matedata(file='Data_Entry_2017_v2020.csv'):
    data = pd.read_csv(file)
    imgs = data['Image Index']
    labels = data['Finding Labels']
    labels = [item.split('|')[0] for item in labels]
    labels = ['Normal' if item=='No Finding' else item for item in labels]
    dataset = {}
    for img, label in zip(list(imgs), labels):
        dataset[img] = label
    return dataset

class MyDataset(Dataset):
    def __init__(
        self, data_dir, train=True, transform=None, target_transform=None, device="cpu", split_file='train_val_list.txt'
    ):
        classes = {
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Effusion': 2,
            'Infiltration': 3,
            'Mass': 4,
            'Nodule': 5,
            'Pneumonia': 6,
            'Pneumothorax': 7,
            'Consolidation': 8,
            'Edema': 9,
            'Emphysema': 10,
            'Fibrosis': 11,
            'Pleural_Thickening': 12,
            'Hernia': 13,
            'Normal': 14
        }
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.split_file = split_file

        dataset = []

        data_dict = prepare_matedata()
        with open(self.split_file, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                dataset.append((line, data_dict[line]))

        if not self.train:
            random.shuffle(dataset)

        self.dataset = dataset
        self.labels = [classes[i[1]] for i in dataset]

        assert len(self.dataset) == len(self.labels)

    def __getitem__(self, index):
        img_path = os.path.join(
            self.data_dir, self.dataset[index][0]
        )
        data = Image.open(img_path)
        data = data.convert('RGB')
        label = self.labels[index]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def getitem(self, index):
        return self.__getitem__(index)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train_data = MyDataset(
        data_dir='../datasets/chestX-ray14/images',
        train=True,
        transform=None,
        device='cuda:0',
        split_file='train_val_list.txt'
    )
