from torch.utils.data import Dataset, DataLoader


class ObjectDataset(Dataset):
    def __init__(self, inputs, outputs):
        super(ObjectDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
