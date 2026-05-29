from torch.utils.data import Dataset

class base_loader(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)