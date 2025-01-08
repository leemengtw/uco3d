from uco3d.data_utils import get_all_load_dataset


def main():
    dataset_root = "/Users/davidnovotny/data/uco3d_examples/"
    dataset = get_all_load_dataset(
        frame_data_builder_kwargs={"dataset_root": dataset_root},
        set_lists_file_name="set_lists_all-categories.sqlite",
    )    
    import pdb; pdb.set_trace()

    
if __name__ == "__main__":
    main()
