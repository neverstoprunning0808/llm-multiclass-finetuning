from datasets import DatasetDict, load_dataset

from config import AppConfig, load_config


def get_data(config: AppConfig):

    # get full dataset
    dataset = load_dataset(config.data.dataset_name)

    # filer for restaurant review only
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    restaurant_train_ds = train_ds.filter(lambda x: "restaurant" in x["text"])
    restaurant_eval_ds = eval_ds.filter(lambda x: "restaurant" in x["text"])

    # filter size:
    restaurant_train_ds_subset = restaurant_train_ds.shuffle(config.data.seed).select(
        range(config.data.train_size)
    )
    restaurant_eval_ds_subset = restaurant_eval_ds.shuffle(config.data.seed).select(
        range(config.data.eval_size)
    )

    # merge together
    merge_ds = {"train": restaurant_train_ds_subset, "val": restaurant_eval_ds_subset}

    # put back to DataseDict
    final_ds = DatasetDict(merge_ds)

    # save to data folder data/restaurant_ds
    final_ds.save_to_disk(config.data.output_path)


if __name__ == "__main__":
    config = load_config()
    get_data(config)
