import logging
import torch
from tqdm import tqdm
from uco3d.data_utils import get_all_load_dataset


logging.basicConfig(level=logging.INFO)


def main():
    dataset_root = "/Users/davidnovotny/data/uco3d_examples/"
    dataset = get_all_load_dataset(
        frame_data_builder_kwargs={
            "dataset_root": dataset_root,
            "load_depths": True,
        },
        dataset_kwargs={"subsets": ["example"]},
        set_lists_file_name="set_lists_all-categories.sqlite",
    )
    
    print("Iterating over dataset")
    for entry in tqdm(dataset):
        pass
    
    print("Randomly iterating over dataset")
    idx = torch.randperm(len(dataset))
    for i in tqdm(idx):
        data = dataset[int(i)]
        pass
    
if __name__ == "__main__":
    main()
