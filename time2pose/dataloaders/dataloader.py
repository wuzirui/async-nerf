from dataloaders.read_default_dataset import DefaultDataset
import torch.utils.data.dataloader as dataloader

def load_dataset(hparams, split='train'):
    dataset = None
    if hparams.dataset_type == 'default':
        dataset = DefaultDataset(hparams, split=split)
    else:
        raise NotImplementedError()
    return dataloader.DataLoader(dataset, shuffle=True, batch_size=hparams.batch_size)