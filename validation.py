from data_set import PromptImagePair
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import argparse

def validate_image_pair_classifier(parquet_file_or_pandas_dataframe: str,
                                   checkout_path: str = None,
                                   device: str = "cuda",
                                   batch_size: int = 1024,
                                   num_workers: int = 9) -> float:
    """
    Validate an image pair classifier using CLIP embeddings.

    This function validates the performance of an image pair classifier model using CLIP embeddings.
    It calculates the accuracy of the classifier on a dataset of paired data, which includes text prompts,
    positive images, and negative images.

    Args:
        parquet_file_or_pandas_dataframe (str or pd.DataFrame): The path to a Parquet file or a Pandas DataFrame
            containing the paired data.
        checkout_path (str, optional): The path to the fine-tuned classifier model checkpoint (default is None).
        device (str, optional): The device to run inference on (e.g., 'cuda' for GPU or 'cpu' for CPU, default is 'cuda').
        batch_size (int, optional): The batch size for data loading (default is 1024).
        num_workers (int, optional): The number of CPU workers for data loading (default is 9).

    Returns:
        float: The accuracy of the image pair classifier on the validation dataset.

    Example:
        >>> accuracy = validate_image_pair_classifier('data.parquet', 'classifier_checkpoint.pth', device='cuda')
        >>> print(f'Accuracy: {accuracy * 100:.2f}%')
    """
    model, preprocess = clip.load("ViT-L/14", "cpu")

    if checkout_path is not None:
        fine_tuned_weights = torch.load(checkout_path)
        model.load_state_dict(fine_tuned_weights)

    device = torch.device(device)
    model = model.to(device)

    accuracy = 0.0
    dataset_shape = 0
    dataset = PromptImagePair(parquet_file_or_pandas_dataframe, preprocess)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    with torch.no_grad():
        for prompts, positives, negatives in tqdm(data_loader):
            prompts_features = model.encode_text(prompts.to(device))
            prompts_features = prompts_features / prompts_features.norm(dim=1, keepdim=True)
            prompts_features = prompts_features.unsqueeze(1)

            positive_features = model.encode_image(positives.to(device))
            positive_features = positive_features / positive_features.norm(dim=1, keepdim=True)
            positive_features = positive_features.unsqueeze(2)

            negative_features = model.encode_image(negatives.to(device))
            negative_features = negative_features / negative_features.norm(dim=1, keepdim=True)
            negative_features = negative_features.unsqueeze(2)

            positive_scores = torch.bmm(prompts_features, positive_features).squeeze(2)
            negative_scores = torch.bmm(prompts_features, negative_features).squeeze(2)

            accuracy += (positive_scores >= negative_scores).sum().item()
            dataset_shape += len(positive_scores)

    accuracy /= dataset_shape
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Validate an image pair classifier using CLIP embeddings.")
    parser.add_argument("parquet_file_or_pandas_dataframe", type=str, help="Path to Parquet file or Pandas DataFrame")
    parser.add_argument("--checkout_path", type=str, default=None, help="Path to classifier checkpoint (default: None)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for data loading (default: 1024)")
    parser.add_argument("--num_workers", type=int, default=9, help="Number of data loading workers (default: 9)")

    args = parser.parse_args()

    accuracy = validate_image_pair_classifier(
        args.parquet_file_or_pandas_dataframe,
        args.checkout_path,
        args.device,
        args.batch_size,
        args.num_workers
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")

# python validation.py test.parquet --checkout_path classifier_checkpoint.pth --device cuda --batch_size 1024 --num_workers 9

if __name__ == "__main__":
    main()