import pandas as pd

from rllm.data.dataset import DatasetRegistry


def prepare_torl_data():
    # Load parquet files from torl_data directory
    train_df = pd.read_parquet("/fsx/zzsamshi/data/torl_data/train.parquet")
    test_df = pd.read_parquet("/fsx/zzsamshi/data/torl_data/test.parquet")
    test_0524_df = pd.read_parquet("/fsx/zzsamshi/data/torl_data/test_0524.parquet")

    def preprocess_fn(row):
        # Extract question from prompt (user message content)
        prompt = row["prompt"]
        question = None
        for msg in prompt:
            if msg["role"] == "user":
                question = msg["content"]
                break

        # Extract ground_truth from reward_model
        ground_truth = row["reward_model"].get("ground_truth", "")

        return {
            "question": question,
            "ground_truth": ground_truth,
            "data_source": row.get("data_source", "torl"),
        }

    # Apply preprocessing
    train_data = [preprocess_fn(row) for _, row in train_df.iterrows()]
    test_data = [preprocess_fn(row) for _, row in test_df.iterrows()]
    test_0524_data = [preprocess_fn(row) for _, row in test_0524_df.iterrows()]

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("torl_math", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("torl_math", test_data, "test")
    test_0524_dataset = DatasetRegistry.register_dataset("torl_math", test_0524_data, "test_0524")

    return train_dataset, test_dataset, test_0524_dataset


if __name__ == "__main__":
    train_dataset, test_dataset, test_0524_dataset = prepare_torl_data()
    print(f"Train dataset: {len(train_dataset)} examples")
    print(train_dataset)
    print(f"\nTest dataset: {len(test_dataset)} examples")
    print(test_dataset)
    print(f"\nTest_0524 dataset: {len(test_0524_dataset)} examples")
    print(test_0524_dataset)

