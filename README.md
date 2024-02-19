**README**

 # Social Reward: Evaluating and Enhancing Generative AI through Million-User Feedback from an Online Creative Community
[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2402.09872)

>**Social Reward: Evaluating and Enhancing Generative AI through Million-User Feedback from an Online Creative Community**<br>
> Arman Isajanyan<sup>1</sup>, Artur Shatveryan<sup>1</sup>, David Kocharyan<sup>1</sup>, Zhangyang Wang<sup>1,2</sup>, Humphrey Shi<sup>1,3</sup> <br>
<sup>1</sup>Picsart AI Research (PAIR), <sup>2</sup>UT Austin, <sup>3</sup>Georgia Tech

>**Abstract**: <br>
> Social reward as a form of community recognition provides a strong source of motivation for users of online platforms to engage and contribute with content. The recent progress of 
  text-conditioned image synthesis has ushered in a collaborative era where AI empowers users to craft original visual artworks seeking community validation. Nevertheless, assessing these models in the context of collective community preference introduces distinct challenges. Existing evaluation methods predominantly center on limited size user studies guided by image quality and prompt alignment. This work pioneers a paradigm shift, unveiling Social Reward - an innovative reward modeling framework that leverages implicit feedback from social network users engaged in creative editing of generated images. We embark on an extensive journey of dataset curation and refinement, drawing from Picsart: an online visual creation and editing platform, yielding a first million-user-scale dataset of implicit human preferences for user-generated visual art named Picsart Image-Social. Our analysis exposes the shortcomings of current metrics in modeling community creative preference of text-to-image models' outputs, compelling us to introduce a novel predictive model explicitly tailored to address these limitations. Rigorous quantitative experiments and user study show that our Social Reward model aligns better with social popularity than existing metrics. Furthermore, we utilize Social Reward to fine-tune text-to-image models, yielding images that are more favored by not only Social Reward, but also other established metrics. These findings highlight the relevance and effectiveness of Social Reward in assessing community appreciation for AI-generated artworks, establishing a closer alignment with users' creative goals: creating popular visual art.

![alt text](assets/comparison.png)

### Setup

   **Setup environment for running train and validation**
   ```bash
   $ git clone https://github.com/Picsart-AI-Research/Social-Reward
   $ cd Social-Reward
   $ python -m venv venv
   $ pip install pip --upgrade
   $ pip install -r requirements.txt
   ```

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
    - `pos_path`: Path to the positive image.
    - `neg_path`: Path to the negative image.

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

### BibTeX

If you use our work in your research, please cite our publication:
```
@misc{isajanyan2024social,
      title={Social Reward: Evaluating and Enhancing Generative AI through Million-User Feedback from an Online Creative Community}, 
      author={Arman Isajanyan and Artur Shatveryan and David Kocharyan and Zhangyang Wang and Humphrey Shi},
      year={2024},
      eprint={2402.09872},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
  
