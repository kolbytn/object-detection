from torch.utils.data import Dataset, DataLoader


class ObjectDataset(Dataset):
    def __init__(self):
        super(ObjectDataset, self).__init__()

    def _preprocess(self):
        pass
