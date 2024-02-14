import torch

class TripletLoss(torch.nn.Module):
    """
    TripletLoss - A PyTorch Module for computing the triplet loss.

    Triplet loss is used for training deep embeddings for tasks like face recognition
    or similarity learning. It encourages the embeddings of anchor-positive pairs
    to be closer than the embeddings of anchor-negative pairs by at least a margin.

    Args:
        margin (float, optional): The margin value (default is 0.3).

    Attributes:
        margin (float): The margin used in the triplet loss calculation.

    Methods:
        forward(anchor, positive, negative):
            Computes the triplet loss for a batch of anchor, positive, and negative samples.

    Examples:
        >>> loss_fn = TripletLoss(margin=0.2)
        >>> anchor = torch.randn(16, 256)  # Embedding for anchor samples
        >>> positive = torch.randn(16, 256)  # Embedding for positive samples
        >>> negative = torch.randn(16, 256)  # Embedding for negative samples
        >>> loss = loss_fn(anchor, positive, negative)
    """
    def __init__(self, margin=0.3):
        """
        Initializes a TripletLoss instance.

        Args:
            margin (float, optional): The margin value (default is 0.3).
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        """
        Calculates the squared Euclidean distance between two tensors.

        Args:
            x1 (torch.Tensor): The first tensor.
            x2 (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The squared Euclidean distance between x1 and x2.
        """
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute the triplet loss for a batch of anchor, positive, and negative samples.

        Args:
            anchor (torch.Tensor): The embeddings of anchor samples.
            positive (torch.Tensor): The embeddings of positive samples.
            negative (torch.Tensor): The embeddings of negative samples.

        Returns:
            torch.Tensor: The triplet loss.
        """
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        losses = losses

        return losses.mean()



