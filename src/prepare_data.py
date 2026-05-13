from datasets import DatasetDict, load_dataset

from config import AppConfig, load_config
from numpy import bincount


def get_data(config: AppConfig):

    # get full dataset
    dataset = load_dataset(config.data.dataset_name)

    # filer for restaurant review only
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    # get only reviews for restaurant, and short-medium lengthy review only
    # to align with the model's pos embedding length ~ 512
    # https://huggingface.co/datasets/Yelp/yelp_review_full
    restaurant_train_ds = train_ds.filter(lambda x: "restaurant" in x["text"].lower() and len(x["text"]) < 500)
    restaurant_eval_ds = eval_ds.filter(lambda x: "restaurant" in x["text"].lower() and len(x["text"]) < 500)

    print("The eval set has: ", len(restaurant_eval_ds), " samples")
    print("The train set has: ", len(restaurant_train_ds), " samples")
    print("The distribution of the train label: ", bincount(restaurant_train_ds["label"]))
    print("The distribution of the eval label: ", bincount(restaurant_eval_ds["label"]))

    # filter size:
    # restaurant_train_ds_subset = restaurant_train_ds.shuffle(config.data.seed).select(
    #     range(config.data.train_size)
    # )
    # restaurant_eval_ds_subset = restaurant_eval_ds.shuffle(config.data.seed).select(
    #     range(config.data.eval_size)
    # )

    # merge together
    # merge_ds = {"train": restaurant_train_ds_subset, "val": restaurant_eval_ds_subset}
    merge_ds = {"train": restaurant_train_ds, "val": restaurant_eval_ds}

    # put back to DataseDict
    final_ds = DatasetDict(merge_ds)

    # save to data folder data/restaurant_ds
    final_ds.save_to_disk(config.data.output_path)


if __name__ == "__main__":
    config = load_config()
    get_data(config)
