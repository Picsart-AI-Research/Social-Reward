import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import clip
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PromptImagePair(Dataset):
    """
    PromptImagePair - A PyTorch Dataset for loading paired data consisting of text prompts and image pairs.

    This dataset is designed for tasks where you have a collection of text prompts paired with positive
    and negative images. It loads the data from a Parquet file or a Pandas DataFrame and applies a
    specified preprocessing function to the images.

    Args:
        parquet_file_or_pandas_dataframe (str or pd.DataFrame): The path to a Parquet file or a Pandas DataFrame
            containing the data. If a string is provided, it's treated as the path to the Parquet file.
        preprocess (callable): A preprocessing function to be applied to the images.

    Methods:
        __getitem__(index):
            Retrieves a single item (text prompt, positive image, negative image) at the given index.
        __len__():
            Returns the total number of items in the dataset.

    Attributes:
        df (pd.DataFrame): The loaded DataFrame containing the data.
        preprocess (callable): The preprocessing function applied to images.

    Examples:
        >>> df = pd.read_parquet('data.parquet')
        >>> preprocess_fn = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
        >>> dataset = PromptImagePair(df, preprocess_fn)
        >>> item = dataset[0]
        >>> text, pos_img, neg_img = item
    """

    def __init__(self, parquet_file_or_pandas_dataframe, preprocess):
        """
        Initializes a PromptImagePair instance.

        Args:
            parquet_file_or_pandas_dataframe (str or pd.DataFrame): The path to a Parquet file or a Pandas DataFrame
                containing the data.
            preprocess (callable): A preprocessing function to be applied to the images.
        """
        self.df = pd.read_parquet(parquet_file_or_pandas_dataframe) if isinstance(parquet_file_or_pandas_dataframe, str) else parquet_file_or_pandas_dataframe
        self.preprocess = preprocess

    def __getitem__(self, index):
        """
        Retrieves a single item (text prompt, positive image, negative image) at the given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing the following:
            - text (torch.Tensor): A tensor representing the tokenized text prompt.
            - pos_img (torch.Tensor): A tensor representing the preprocessed positive image.
            - neg_img (torch.Tensor): A tensor representing the preprocessed negative image.
        """
        items = self.df.iloc[index]
        prompt = items.prompt
        pos_path = items.pos_path
        neg_path = items.neg_path

        pos_img = Image.open(pos_path)
        neg_img = Image.open(neg_path)
        text = clip.tokenize([prompt], truncate=True)[0]

        pos_img = self.preprocess(pos_img)
        neg_img = self.preprocess(neg_img)


        return text, pos_img, neg_img

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.df)
