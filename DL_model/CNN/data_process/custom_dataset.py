class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index].clone().detach()
        feature = feature.to(dtype=torch.float32)
    
        label = self.labels[index].clone().detach()
        label = label.to(dtype=torch.int64)
        return feature, label
