import clip
from torch.utils.data import DataLoader
from data_set import PromptImagePair
from losses import *
from torch import optim
from tqdm import tqdm
from accelerate import Accelerator
import os
import argparse



def finetune_model(training_file,
                   training_mode="all",
                   batch_size = 128,
                   n_epochs=5,
                   save_folder = "./clip_finetune",
                   loss_name="triplet",
                   checkout_path=None,
                   learning_rate = 0.00003):
    """
    Fine-tune a CLIP model for a specific training task.

    This function fine-tunes a CLIP model for a given training task using the specified training mode and loss function.

    Args:
        training_file (str): The path to a Parquet file containing training data.
        training_mode (str, optional): The training mode, which determines which parts of the model are trained (default: "all").
        batch_size (int, optional): The batch size for training (default: 128).
        n_epochs (int, optional): The number of training epochs (default: 5).
        save_folder (str, optional): The folder where fine-tuned model checkpoints will be saved (default: "./clip_finetune").
        loss_name (str, optional): The loss function to use for training (default: "triplet").
        checkout_path (str, optional): The path to a pre-trained model checkpoint for fine-tuning (default: None).
        learning_rate (float, optional): The learning rate for optimization (default: 0.00003).

    Returns:
        None

    Example:
        >>> finetune_model(
        >>>     training_file='train_data.parquet',
        >>>     training_mode='all',
        >>>     batch_size=128,
        >>>     n_epochs=10,
        >>>     save_folder='./fine_tuned_models',
        >>>     loss_name='triplet',
        >>>     checkout_path='pretrained_model.pth',
        >>>     learning_rate=0.0001
        >>> )
    """

    os.makedirs(save_folder, exist_ok=True)

    accelerator = Accelerator()
    device = accelerator.device

    print(checkout_path)
    model, preprocess = clip.load("ViT-L/14", "cpu")
    if checkout_path is not None:
        fined_tuned_weights = torch.load(checkout_path,map_location="cpu")
        model.load_state_dict(fined_tuned_weights)

    model = model.to(device)


    train_dataset = PromptImagePair(training_file, preprocess=preprocess)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    if loss_name == "triplet":
        training_loss = TripletLoss(margin=0.3)
    else:
        raise Exception(f"Wrong loss {loss_name}")

    if training_mode == "all":
        model.train()
        model.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    elif training_mode == "visual":
        model.eval()
        model.requires_grad = False

        model.visual.train()
        model.visual.requires_grad = True

        optimizer = optim.Adam(model.visual.parameters(), lr=learning_rate)
    elif training_mode == "visual_upper_layers":

        model = model.eval()
        model.requires_grad = False


        upper_layers = model.visual.transformer.resblocks[21:]
        layer_norm = model.visual.ln_post

        upper_layers.train()
        layer_norm.train()

        upper_layers.requires_grad = True
        layer_norm.requires_grad = True
        optimizer = optim.Adam(list(upper_layers.parameters()) + list(layer_norm.parameters()), lr=learning_rate)

    elif training_mode == "visual_upper_layers_textual_upper_layers":

        model = model.eval()
        model.requires_grad = False


        upper_layers = model.visual.transformer.resblocks[21:]
        layer_norm = model.visual.ln_post

        upper_layers.train()
        layer_norm.train()

        upper_layers.requires_grad = True
        layer_norm.requires_grad = True


        textual_upper_layers = model.transformer.resblocks[10:]
        textual_ln_final = model.ln_final
        textual_projection = model.text_projection

        textual_upper_layers.train()
        textual_ln_final.train()

        textual_upper_layers.requires_grad = True
        textual_ln_final.requires_grad = True
        textual_projection.requires_grad = True

        optimizer = optim.AdamW(list(upper_layers.parameters()) + list(layer_norm.parameters()) + list(textual_upper_layers.parameters())  + list(textual_ln_final.parameters())  + [textual_projection], lr=learning_rate)
    elif training_mode == "visual_upper_layers_textual_upper_layers_deeper":

        model = model.eval()
        model.requires_grad = False


        upper_layers = model.visual.transformer.resblocks[20:]
        layer_norm = model.visual.ln_post

        upper_layers.train()
        layer_norm.train()

        upper_layers.requires_grad = True
        layer_norm.requires_grad = True


        textual_upper_layers = model.transformer.resblocks[9:]
        textual_ln_final = model.ln_final
        textual_projection = model.text_projection

        textual_upper_layers.train()
        textual_ln_final.train()
        #textual_projection.train()


        textual_upper_layers.requires_grad = True
        textual_ln_final.requires_grad = True
        textual_projection.requires_grad = True

        optimizer = optim.AdamW(list(upper_layers.parameters()) + list(layer_norm.parameters()) + list(textual_upper_layers.parameters())  + list(textual_ln_final.parameters())  + [textual_projection], lr=learning_rate)


    elif training_mode == "visual_last_layer":

        model = model.eval()
        model.requires_grad = False


        upper_layers = model.visual.transformer.resblocks[23:]
        layer_norm = model.visual.ln_post

        upper_layers.train()
        layer_norm.train()

        upper_layers.requires_grad = True
        layer_norm.requires_grad = True
        optimizer = optim.Adam(list(upper_layers.parameters()) + list(layer_norm.parameters()), lr=learning_rate)
    else:
        raise Exception(f"Wrong training mode: {training_mode}")

    model_encode_text, model_encode_image,  optimizer, train_dataloader = accelerator.prepare(model.encode_text, model.encode_image, optimizer, train_dataloader)

    progress_bar = tqdm(range(n_epochs*len(train_dataloader)), disable=not accelerator.is_local_main_process)


    global_step = 0
    for epoch in range(n_epochs):
        for prompts, positives, negatives in train_dataloader:
            with accelerator.accumulate(model):
                prompts_features = model_encode_text(prompts)
                prompts_features = prompts_features / prompts_features.norm(dim=1, keepdim=True)

                positive_features = model_encode_image(positives)
                positive_features = positive_features / positive_features.norm(dim=1, keepdim=True)

                negative_features = model_encode_image(negatives)
                negative_features = negative_features / negative_features.norm(dim=1, keepdim=True)

                optimizer.zero_grad()
                loss = training_loss(anchor=prompts_features, positive=positive_features,
                                     negative=negative_features)

                accelerator.backward(loss)
                optimizer.step()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    print(f"Epoch: {epoch}, Global step: {global_step} Loss: {loss.item()}")
                    if global_step % 800 == 0 and global_step != 0:
                        if accelerator.is_main_process:
                            torch.save(model.state_dict(),os.path.join(save_folder,f"clip_epoch_{epoch}_global_step_{global_step}.pth"))

        if accelerator.is_main_process:
            torch.save(model.state_dict(),
                       os.path.join(save_folder, f"clip_epoch_{epoch}_global_step_final.pth"))
            torch.save(model.state_dict(),
                       os.path.join(save_folder, f"clip_epoch_{epoch}_global_step_{global_step}.pth"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune CLIP model.')

    parser.add_argument('--training_file', type=str, help='Path to the training file')
    parser.add_argument('--training_mode', type=str, help='Training mode')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--n_epochs', type=int, help='Number of epochs')
    parser.add_argument('--save_folder', type=str, help='Path to save folder')

    parser.add_argument('--loss_name', type=str, default='triplet', help='Name of the loss (default: triplet)')
    parser.add_argument('--checkout_path', type=str, default=None, help='Checkout path (default: None)')
    parser.add_argument('--learning_rate', type=float, default=0.00003, help='Learning rate (default: 0.00003)')

    args = parser.parse_args()

    training_file = args.training_file
    training_mode = args.training_mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    save_folder = args.save_folder
    loss_name = args.loss_name
    checkout_path = args.checkout_path
    learning_rate = args.learning_rate
    print(args)

    finetune_model(
        training_file=training_file,
        training_mode=training_mode,
        batch_size=batch_size,
        n_epochs=n_epochs,
        save_folder=save_folder,
        loss_name=loss_name,
        checkout_path=checkout_path,
        learning_rate=learning_rate
    )
