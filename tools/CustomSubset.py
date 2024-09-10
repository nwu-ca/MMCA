
class CustomSubset(MultiviewImgDataset, torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(root_dir=dataset.root_dir, scale_aug=dataset.scale_aug, rot_aug=dataset.rot_aug,
                         test_mode=dataset.test_mode, num_models=dataset.num_models, num_views=dataset.num_views)

        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.indices)