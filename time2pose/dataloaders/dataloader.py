from dataloaders.read_standard_dataset import read_data

def load_dataset(datapath, dataset_type):
    if dataset_type == 'kitti':
        pose = read_data(datapath)
    else:
        raise NotImplementedError()
    return pose