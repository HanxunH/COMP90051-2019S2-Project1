import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_start_method('spawn', force=True)


class ProjectDataset(Dataset):
    def __init__(self, file_path, idx_file_path, num_of_classes):
        with open(idx_file_path, 'rb') as handle:
            self.class_idx = pickle.load(handle)
            self.idx_class = dict((y, x) for x, y in self.class_idx.items())

        self.num_of_classes = num_of_classes
        self.samples_frame = pd.read_csv(file_path, sep='\t')
        self.labels = []
        self.sentences = []
        # with open(file_path, encoding='utf-8') as file:
        #     reader = csv.reader((x.replace('\0', '') for x in file), delimiter='\t')
        #     for i, row in enumerate(reader):
        #         id = int(row[0])
        #         instance = row[1]
        #         indices = torch.tensor(self.class_idx[id])
        #         one_hot_id = torch.nn.functional.one_hot(indices, num_of_classes)
        #         self.labels.append(one_hot_id)
        #         self.sentences.append(instance)
        # print("Total instance: %d" % (len(self.sentences)))
        return

    def __len__(self):
        return len(self.samples_frame)

    def get_one_hot(self, id):
        indices = torch.tensor(self.class_idx[id])
        one_hot_id = torch.nn.functional.one_hot(indices, self.num_of_classes)
        return torch.FloatTensor(one_hot_id.unsqueeze(0))

    def __getitem__(self, idx):
        id, sentnece = self.samples_frame.iloc[idx, 0:].values
        indices = torch.tensor(self.class_idx[id])
        one_hot_id = torch.nn.functional.one_hot(indices, self.num_of_classes)
        return sentnece, one_hot_id


if __name__ == '__main__':
    train_set = ProjectDataset(file_path="data/v1/train_set_v1.txt",
                               idx_file_path="data/v1/v1_idx.pickle",
                               num_of_classes=9292)
    print(len(train_set))
    # dev_set = ProjectDataset("data/v1/dev_dataloader")
    train_loaders = DataLoader(train_set,
                               batch_size=1,
                               shuffle=False,
                               pin_memory=True,
                               num_workers=1)

    for i, (embed, labels) in enumerate(train_loaders):
        print(embed, labels)
        continue
