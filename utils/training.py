from torch.optim.lr_scheduler import *
import math
import torch
import time
import os
from sklearn.metrics import f1_score , balanced_accuracy_score
from .gradient_attack import GradientAttack 
import numpy as np
from torch.utils.data import DataLoader
import shutil
from torch.utils.tensorboard import SummaryWriter
import subprocess

def update_dataloader(dataset, new_batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=new_batch_size, shuffle=shuffle)

def train_model(
        model,
        train_loader=None,
        val_loader=None,
        criterion=None,
        optimizer_class=None,
        optimizer_config=None,
        scheduler_class=None,
        scheduler_config=None,
        window_len=None,
        original_val_labels=None,
        n_epochs=500,
        patience=200,
        device="cuda",
        save_model_checkpoints=True,
        save_path="DeepConvNetMI",
        domain_lambda=0.1,
        lambda_scheduler_fn=None,
        scheduler_fn=None,
        adversarial_training=True,
        adversarial_steps=10,
        adversarial_epsilon=0.1,
        adversarial_alpha=0.01,
        adversarial_factor=0.5,
        update_loader=None,
        n_classes=None,
        save_best_only=False,
        tensorboard = False,
        ):
    
    """
    Trains a deep EEG model with optional domain adaptation and adversarial training.

    Parameters:
    ----------
    model : torch.nn.Module
        Your EEG classification model. Should return (class_logits, domain_logits).

    train_loader : DataLoader, optional
        DataLoader for training data. Each batch should return (X, weights, labels, domain_labels).
        If None, training will not proceed.

    val_loader : DataLoader, optional
        DataLoader for validation data. Same format as train_loader but used for validation only.

    criterion : torch.nn loss function
        Loss function used for both label and domain predictions.

    optimizer_class : torch.optim class
        e.g. `torch.optim.Adam`. Used to initialize the optimizer.

    optimizer_config : dict
        Keyword arguments passed to the optimizer (e.g., learning rate, weight decay).

    scheduler_class : torch.optim.lr_scheduler class, optional
        e.g. `torch.optim.lr_scheduler.ReduceLROnPlateau`. If provided, will be used to adjust learning rate.

    scheduler_config : dict, optional
        Config passed to the LR scheduler class.

    window_len : int
        Number of windows per original trial. Used for validation aggregation.

    original_val_labels : torch.Tensor
        Labels corresponding to the original, unwindowed validation trials.

    n_epochs : int
        Maximum number of epochs to train. Default is 500.

    patience : int
        How many epochs to wait for improvement before early stopping.

    device : str or torch.device
        Device to use: e.g., 'cuda' or 'cpu'.

    save_model_checkpoints : bool
        If True, saves checkpoints during training.

    save_path : str
        Directory path to store saved models and logs.

    domain_lambda : float
        Initial weight for the domain loss (for domain adaptation).

    lambda_scheduler_fn : callable, optional
        Function to dynamically update `domain_lambda` each epoch. Signature: lambda old_lambda, epoch → new_lambda.

    scheduler_fn : callable, optional
        Custom function to update adversarial parameters dynamically.
        Should return updated values for (domain_lambda, adversarial_training, adversarial_steps, adversarial_epsilon, adversarial_alpha, adversarial_factor).

    adversarial_training : bool
        If True, enables FGSM/PGD-like adversarial training during model optimization.

    adversarial_steps : int
        Number of steps for gradient attack (if enabled). More steps → stronger adversarial examples.

    adversarial_epsilon : float
        Maximum perturbation allowed for adversarial attack.

    adversarial_alpha : float
        Step size for iterative adversarial attack (PGD-style).

    adversarial_factor : float
        Weighting factor for adversarial loss in total loss computation.

    update_loader : tuple (epoch_index, new_batch_size), optional
        If specified, updates the training loader to a new batch size after `epoch_index`.

    n_classes : int
        Number of output classes. Required for initializing validation tensors.

    save_best_only : bool
        If True, only saves the checkpoint with the best validation performance (based on F1 + balanced accuracy).

    tensorboard : bool
        If True, initializes TensorBoard logging for losses, metrics, and layer introspection.

    Returns:
    -------
    float
        Best validation metric (average of F1 and balanced accuracy) seen during training.
    """

    print(f"Total Parameters of the Model ======> {sum(p.numel() for p in model.parameters()):,} paramaters")
    if tensorboard:
        log_dir = os.path.join(save_path, "logs")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        command = [
            "tensorboard",
            "--logdir", log_dir,
            "--port", str(6006),
            "--reload_interval", "1"
        ]

        tensorboard_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Launching TensorBoard at http://localhost:{6006}/ ...")
    

    if update_loader:
        update_epoch , batch_size = update_loader
    if isinstance(device,str):
        device = torch.device(device)
    if save_path:
        if os.path.isdir(save_path):
            print("Path Exists. Contents of this folder will be modified save_path is : ",save_path)
    else:
        print("Making a new directory at : ", save_path , " Checkpoints will be saved there. ")
        os.makedirs(save_path, exist_ok=True) 
    domain_lambda_ = domain_lambda
    best_val_metric = float('-inf') # Store balanced accuracy
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = -1 # To track the epoch of the best model
        # --- 4. Training Loop ---
    print("--- Starting Training Loop ---")

    optimizer = optimizer_class(model.parameters(), **optimizer_config)


    scheduler = None
    if scheduler_class is not None and scheduler_config is not None:
        scheduler = scheduler_class(optimizer, **scheduler_config)
    for epoch in range(n_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        all_preds_train = []
        all_targets_train = []
        all_preds_train_adv=[]
        start_time = time.time()

        # if 'scheduler' in locals() and scheduler is not None:
        #     scheduler.step() # Step the scheduler after optimizer.step() if you want per-epoch updates
        if lambda_scheduler_fn:
                domain_lambda_ = lambda_scheduler_fn(domain_lambda , epoch)

        if epoch==update_epoch:
            train_loader = update_dataloader(train_loader.dataset,new_batch_size=batch_size)


        if scheduler_fn:
            domain_lambda, adversarial_training, adversarial_steps, adversarial_epsilon, adversarial_alpha, adversarial_factor = scheduler_fn(
                epoch + 1
            )

        for batch_x, weights_batch, batch_y,subject_label in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.long)
            weights_batch = weights_batch.to(device, dtype=torch.float32)
            subject_label = subject_label.to(device,dtype=torch.long)

            if adversarial_training:
                model.eval()#Freezes model for attack generation (because we don't want to change model weights at this stage)
                batch_x_adv = GradientAttack(model, batch_x, batch_y, criterion,
                            alpha=adversarial_alpha,
                            epsilon=adversarial_epsilon,
                            steps=adversarial_steps,
                            clamp_min=-16.8080,
                            clamp_max=16.8080)
                model.train()
            optimizer.zero_grad()

            outputs_labels,outputs_domain = model(batch_x,domain_lambda_)
            if adversarial_training:
                outputs_adv_labels,_ = model(batch_x_adv,domain_lambda_)

            label_loss= criterion(outputs_labels, batch_y) 
            domain_loss = criterion(outputs_domain,subject_label)
            if adversarial_training:
                adv_loss = criterion(outputs_adv_labels,batch_y)
            else:
                adv_loss=torch.tensor(0.0)

            label_loss = (label_loss * weights_batch).mean() # Re-enable weighted loss!
            domain_loss = domain_loss.mean()
            adv_loss = adv_loss.mean()

            total_loss = label_loss  +   domain_loss   +    adversarial_factor * adv_loss

            total_loss.backward()

            optimizer.step()
            epoch_loss += total_loss.item() * batch_x.size(0)

            preds = torch.argmax(outputs_labels, dim=1)
            if adversarial_training:
                preds_adv = torch.argmax(outputs_adv_labels, dim=1)


            if adversarial_training:
                all_preds_train_adv.extend(preds_adv.cpu().numpy())
            all_preds_train.extend(preds.cpu().numpy())
            all_targets_train.extend(batch_y.cpu().numpy())

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        # Use average='weighted' for F1-score in multi-class classification
        train_f1 = f1_score(all_targets_train, all_preds_train, average='weighted')
        train_adv_f1 = f1_score(all_targets_train, all_preds_train_adv, average='weighted') if all_preds_train_adv else 0
        # 📉 Validation
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for val_x, weights_val_batch, val_y_windowed,_ in val_loader:
                val_x = val_x.to(device, dtype=torch.float32)
                val_y_windowed = val_y_windowed.to(device, dtype=torch.long) # Not directly used for loss, but kept for consistency
                weights_val_batch = weights_val_batch.to(device, dtype=torch.float32)

                val_outputs_raw, domain_outputs = model(val_x,domain_lambda_)


                num_original_trials_val = original_val_labels.shape[0]
                aggregated_outputs = torch.zeros((num_original_trials_val, n_classes), device=device, dtype=torch.float32)
                
                k = 0 # Counter for original trials
                for i in range(0, val_outputs_raw.shape[0], int(window_len)):
                    logits = val_outputs_raw[i : i + int(window_len)] # [WINDOW_LEN_GLOBAL, n_classes]
                    w_cur = weights_val_batch[i : i + int(window_len)].unsqueeze(1) # [WINDOW_LEN_GLOBAL, 1]
                    # Re-enable weighted aggregation for validation!
                    denom = w_cur.sum()
                    if denom > 1e-8: # Add small epsilon to avoid division by zero
                        aggregated_outputs[k] = (logits * w_cur).sum(dim=0) / denom
                    else:
                        # Fallback if weights are all zero for a trial's windows
                        aggregated_outputs[k] = logits.mean(dim=0) # Use unweighted mean as fallback
                    
                    k += 1
                
                
                # Loss for aggregated outputs against original labels
                loss = criterion(aggregated_outputs, original_val_labels.to(device)).mean() # Use mean reduction for aggregated loss
                val_loss += loss.item() * aggregated_outputs.size(0)

                preds_agg = torch.argmax(aggregated_outputs, dim=1)
                val_preds.extend(preds_agg.cpu().numpy())
                val_targets.extend(original_val_labels.cpu().numpy()) # Use original labels for
            
        avg_val_loss = val_loss / len(val_preds) # Use number of aggregated predictions for average loss
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        early_stopping_criteria = (val_f1+val_bal_acc)/2


        if tensorboard:
            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
            writer.add_scalar("F1/train", train_f1, epoch + 1)
            writer.add_scalar("F1/train_adv", train_adv_f1, epoch + 1)
            writer.add_scalar("F1/val", val_f1, epoch + 1)
            writer.add_scalar("BalancedAccuracy/val", val_bal_acc, epoch + 1)
            writer.add_scalar("Lambda/domain", domain_lambda, epoch + 1)
            writer.add_scalar("Lambda/domain_loss", domain_loss, epoch + 1)
            writer.add_scalar("Adversarial/steps", adversarial_steps, epoch + 1)
            writer.add_scalar("Adversarial/epsilon", adversarial_epsilon, epoch + 1)
            writer.add_scalar("Adversarial/alpha", adversarial_alpha, epoch + 1)
            writer.add_scalar("Adversarial/factor", adversarial_factor, epoch + 1)

            depth = 0
            scale = 100

            for name, module in model.named_modules():
                if hasattr(module, 'last_gates'):
                    depth += 1
                    for branch_idx, gate in enumerate(module.last_gates):
                        # gate: [B, C, 1]
                        gate_curve = gate[0, :, 0].detach().cpu()  # [C]

                        C = gate_curve.size(0)
                        counts = (gate_curve * scale).round().long().clamp(min=1)  # [C]
                        channel_idxs = torch.arange(C).repeat_interleave(counts)   # [sum(counts)]

                        writer.add_histogram(
                            f"Gates/Branch{branch_idx+1}_Depth{depth}_ChBiasHist",
                            channel_idxs.numpy(),
                            global_step=epoch,
                            bins=C
                        )
                if hasattr(module, 'temporal_weights'):
                    temp_curve = module.temporal_weights[29*6+5, 0].detach().cpu()  # [T]
                    T = temp_curve.size(0)
                    scale = 100  # controls how many samples per "bin" based on intensity
                    counts = (temp_curve * scale).round().long().clamp(min=1)  # [T]
                    time_idxs = torch.arange(T).repeat_interleave(counts)      # [sum(counts)]

                    writer.add_histogram(
                        f"TemporalWeights/{name}_TimeSeriesHist",
                        time_idxs.numpy(),
                        global_step=epoch,
                        bins=T
                    )
                    writer.flush()

        # --- 6. Checkpointing and Early Stopping ---
        # Save checkpoint every epoch for robust recovery, or modify to save less frequently

        if scheduler is not None:
            # Check the type of scheduler to step correctly
            if isinstance(scheduler, ReduceLROnPlateau):
                # For ReduceLROnPlateau, step with the validation metric you are monitoring
                scheduler.step(val_bal_acc)
            else:
                scheduler.step()

        if save_path:
            if save_model_checkpoints:
                if save_best_only:
                    if early_stopping_criteria > best_val_metric:
                        save_path_for_best_version = os.path.join(save_path, "best_model_.pth")
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_f1": val_f1,
                                "val_bal_acc": val_bal_acc
                            },
                            save_path_for_best_version
                        )
                        print("✅ Best checkpoint updated (save_best_only=True). at ",save_path)
                else:
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_f1": val_f1,
                            "val_bal_acc": val_bal_acc
                        },
                        os.path.join(save_path, f"checkpoint_epoch_{epoch+1:03d}.pth")
                    )
                    print(f"🟢 Checkpoint saved for epoch {epoch+1}")

        # Early stopping logic: Use val_bal_acc as the primary metric
        if early_stopping_criteria > best_val_metric:
            best_val_metric = early_stopping_criteria
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            print("🟢 Valid" "ation Balanced Accuracy improved. Saving best model state...")
        else:
            epochs_without_improvement += 1
            print(f"🟡 No improvement for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= patience:
            print(f"🔴 Early stopping triggered after {epoch+1} epochs. Best Balanced Accuracy: {best_val_metric:.4f} at epoch {best_epoch}")
            break

        # --- 7. Logging ---
        print(f"Epoch [{epoch+1}/{n_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f} | adversarial F1 : {train_adv_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f} | "
            f"Time: {time.time() - start_time:.1f}s | "
            f"Balanced Accuracy Val: {val_bal_acc:.4f} | - "
            f"Domain Loss: {domain_loss:.4f}")
        if epoch==100 and best_val_metric<0.5:
            print("---BREAKING TRAINING----")
            return best_val_metric
    # --- 8. Final Actions After Training ---
    print("\n--- Training Finished ---")
    if best_model_state:
        print(f"Best validation Balanced Accuracy: {best_val_metric:.4f} achieved at Epoch {best_epoch}")
    writer.close()
    return best_val_metric


    

import torch.nn.functional as F

def predict_optimized(
        model,
        windows_per_trial=None,
        loader=None,
        probability=False,
        device="cpu",
        ):
    

    """
    A lightweight prediction function that turns window-level EEG inputs into trial-level outputs.

    It's meant for fast inference — for example, during test-time when everything's already preloaded. 
    You pass in a model and a loader (or a direct tuple of windows and weights), and it handles 
    the rest: moves data to the correct device, runs the model, and averages predictions across all windows 
    for each trial (if windows_per_trial is set).

    Args:
        model (torch.nn.Module): The trained model you want to use for prediction. Should support forward(x, place_holder_float).
        windows_per_trial (int, optional): How many windows make up one trial (for averaging predictions). 
        loader (DataLoader or Tuple): A PyTorch DataLoader or a tuple like (x, weights). Can be used for test data.
        probability (bool): If True, returns softmax probabilities. If False, returns hard class labels.
        device (str): The device to run the model on. "cpu" or "cuda".

    Returns:
        np.ndarray: Predicted labels or class probabilities for each trial (or window if windows_per_trial=None).
    """
    
    model.eval()
    with torch.no_grad():

        if isinstance(loader , torch.utils.data.DataLoader):
            x = loader.dataset.data.detach().clone().to(device, dtype=torch.float32)
            weights = loader.dataset.weigths.detach().clone().to(device, dtype=torch.float32)
        else:
            x , weights = loader
        outputs , _ = model(x,0)
        weighted_outputs = (outputs * weights.unsqueeze(1)).T.unsqueeze(0)  ##[n_windows , n_classes] ===> [1 , n_classes , n_windows]##

        trial_level_preds =F.avg_pool1d(weighted_outputs, kernel_size=windows_per_trial, stride=windows_per_trial)
        trial_level_preds = trial_level_preds.squeeze(0).T

        probs = torch.softmax(trial_level_preds, dim=1)
        preds = torch.argmax(probs, dim=1)

    return (probs.cpu().numpy() if probability else preds.cpu().numpy())
