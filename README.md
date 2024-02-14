**README**

This repository contains code for fine-tuning and validating a Contrastive Learning of Visual Representations (CLIP) model for a specific training task involving paired data consisting of text prompts and image pairs. The primary goal is to train a model for predicting social reward, specifically identifying which of two images is remixable. The code utilizes PyTorch and the CLIP library for training and validation.

### Files and Structure

1. **PromptImagePair Dataset (`data_set.py`):**
    - PyTorch Dataset for loading paired data (text prompts, positive images, negative images).
    - Supports loading data from Parquet files or Pandas DataFrame.
    - Applies a specified preprocessing function to the images.

    **Example:**
    ```python
    df = pd.read_parquet('data.parquet')
    preprocess_fn = torchvision.transforms.Compose([torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
    dataset = PromptImagePair(df, preprocess_fn)
    item = dataset[0]
    text, pos_img, neg_img = item
    ```

2. **Triplet Loss Module (`losses.py`):**
    - PyTorch module for computing the triplet loss.
    - Used for training deep embeddings for similarity learning.

    **Example:**
    ```python
    loss_fn = TripletLoss(margin=0.2)
    anchor = torch.randn(16, 256)
    positive = torch.randn(16, 256)
    negative = torch.randn(16, 256)
    loss = loss_fn(anchor, positive, negative)
    ```
### Running the Example

1. **Data Preparation:**
    - Prepare a Parquet file (`train_data.parquet`) containing paired data (prompt, positive image path, negative image path).
    - Similarly, prepare a validation Parquet file (`validation_data.parquet`).

    **Example Parquet File Structure (`train_data.parquet` and `validation_data.parquet`):**

    | prompt     | pos_path                             | neg_path                             |
    |------------|--------------------------------------|--------------------------------------|
    | "Prompt 1" | "/path/to/remixable/image1.jpg"      | "/path/to/non-remixable/image2.jpg"  |
    | "Prompt 2" | "/path/to/remixable/image3.jpg"      | "/path/to/non-remixable/image4.jpg"  |
    | ...        | ...                                  | ...                                  |

    - `prompt`: Text prompt corresponding to each pair of positive and negative images.
    - `pos_path`: Path to the image considered remixable.
    - `neg_path`: Path to the image considered non-remixable.

2. **Training:**
    - Fine-tune the CLIP model using the training script.
    ```bash
   accelerate launch\
    train_pair_pos_neg.py\
    --training_file training_file.parquet\
    --training_mode visual_upper_layers_textual_upper_layers\
    --batch_size 32\
    --n_epochs 10\
    --save_folder ./clip_model\
    --loss_name triplet
    ```

3. **Validation:**
    - Validate the fine-tuned model using the validation script.
    ```bash
    python validate.py 'validation_data.parquet' --checkout_path 'classifier_checkpoint.pth' --device 'cuda' --batch_size 1024 --num_workers 9
    ```

**Note:**
- Ensure all required dependencies (PyTorch, CLIP, tqdm, etc.) are installed.
- Adjust file paths and parameters according to your dataset and preferences.
