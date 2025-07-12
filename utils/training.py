from torch.optim.lr_scheduler import *
import math
import torch
import time
import os
from sklearn.metrics import f1_score , balanced_accuracy_score
from .gradient_attack import GradientAttack #Function for gradient attack
from .rank_ensemble import RankAveragingEnsemble
import numpy as np



from torch.utils.tensorboard import SummaryWriter


def train_model(
    model,
    train_loader=None,
    val_loader=None,
    criterion=None,
    optimizer=None,
    window_len=None,
    original_val_labels=None,
    n_epochs=500,
    patience=200,
    device="cuda",
    save_model_checkpoints=True,
    save_path="DeepConvNetMI",
    lr_scheduler=None,
    scheduler_fn=None,
    domain_lambda=0.1,
    adversarial_training=True,
    adversarial_steps=10,
    adversarial_epsilon=0.1,
    adversarial_alpha=0.01,
    adversarial_factor=0.5,
    n_classes=None,
    save_best_only=False,
):


    writer = SummaryWriter(log_dir=os.path.join(save_path, "logs"))

    if isinstance(device, str):
        device = torch.device(device)

    os.makedirs(save_path, exist_ok=True)

    best_val_metric = float("-inf")
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = -1

    print("--- Starting Training Loop ---")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        all_preds_train = []
        all_targets_train = []
        all_preds_train_adv = []
        start_time = time.time()

        if scheduler_fn:
            domain_lambda, adversarial_training, adversarial_steps, adversarial_epsilon, adversarial_alpha, adversarial_factor = scheduler_fn(
                epoch + 1
            )

        for batch_x, weights_batch, batch_y, subject_label in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.long)
            weights_batch = weights_batch.to(device, dtype=torch.float32)
            subject_label = subject_label.to(device, dtype=torch.long)

            if adversarial_training:
                model.eval()
                batch_x_adv = GradientAttack(
                    model,
                    batch_x,
                    batch_y,
                    criterion,
                    alpha=adversarial_alpha,
                    epsilon=adversarial_epsilon,
                    steps=adversarial_steps,
                    clamp_min=-16.8080,
                    clamp_max=16.8080,
                )
                model.train()

            optimizer.zero_grad()

            outputs_labels, outputs_domain = model(batch_x, domain_lambda)
            if adversarial_training:
                outputs_adv_labels, _ = model(batch_x_adv, domain_lambda)

            label_loss = criterion(outputs_labels, batch_y)
            domain_loss = criterion(outputs_domain, subject_label)
            adv_loss = criterion(outputs_adv_labels, batch_y) if adversarial_training else torch.tensor(0.0)

            label_loss = (label_loss * weights_batch).mean()
            domain_loss = domain_loss.mean()
            adv_loss = adv_loss.mean()

            total_loss = label_loss + domain_loss + adversarial_factor * adv_loss
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
        train_f1 = f1_score(all_targets_train, all_preds_train, average="weighted")
        train_adv_f1 = (
            f1_score(all_targets_train, all_preds_train_adv, average="weighted")
            if all_preds_train_adv
            else 0
        )

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for val_x, weights_val_batch, val_y_windowed, _ in val_loader:
                val_x = val_x.to(device, dtype=torch.float32)
                val_y_windowed = val_y_windowed.to(device, dtype=torch.long)
                weights_val_batch = weights_val_batch.to(device, dtype=torch.float32)

                val_outputs_raw, domain_outputs = model(val_x, domain_lambda)
                num_original_trials_val = original_val_labels.shape[0]
                aggregated_outputs = torch.zeros((num_original_trials_val, n_classes), device=device)

                k = 0
                for i in range(0, val_outputs_raw.shape[0], int(window_len)):
                    logits = val_outputs_raw[i : i + int(window_len)]
                    w_cur = weights_val_batch[i : i + int(window_len)].unsqueeze(1)
                    denom = w_cur.sum()
                    if denom > 1e-8:
                        aggregated_outputs[k] = (logits * w_cur).sum(dim=0) / denom
                    else:
                        aggregated_outputs[k] = logits.mean(dim=0)
                    k += 1

                loss = criterion(aggregated_outputs, original_val_labels.to(device)).mean()
                val_loss += loss.item() * aggregated_outputs.size(0)
                preds_agg = torch.argmax(aggregated_outputs, dim=1)
                val_preds.extend(preds_agg.cpu().numpy())
                val_targets.extend(original_val_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")
        val_bal_acc = balanced_accuracy_score(val_targets, val_preds)

        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
        writer.add_scalar("F1/train", train_f1, epoch + 1)
        writer.add_scalar("F1/train_adv", train_adv_f1, epoch + 1)
        writer.add_scalar("F1/val", val_f1, epoch + 1)
        writer.add_scalar("BalancedAccuracy/val", val_bal_acc, epoch + 1)
        writer.add_scalar("Lambda/domain", domain_lambda, epoch + 1)
        writer.add_scalar("Lambda/domain_loss", domain_lambda, epoch + 1)
        writer.add_scalar("Adversarial/steps", adversarial_steps, epoch + 1)
        writer.add_scalar("Adversarial/epsilon", adversarial_epsilon, epoch + 1)
        writer.add_scalar("Adversarial/alpha", adversarial_alpha, epoch + 1)
        writer.add_scalar("Adversarial/factor", adversarial_factor, epoch + 1)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)


        if lr_scheduler:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_bal_acc)
            else:
                lr_scheduler.step()

        if save_model_checkpoints:
            if save_best_only:
                if val_bal_acc > best_val_metric:
                    save_path_for_best_version = os.path.join(save_path, "best_model_.pth")

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_f1": val_f1,
                            "val_bal_acc": val_bal_acc,
                        },
                        save_path_for_best_version,
                    )
                    print("✅ Best checkpoint updated (save_best_only=True). at ",save_path)

            else:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_f1": val_f1,
                        "val_bal_acc": val_bal_acc,
                    },
                    os.path.join(save_path, f"checkpoint_epoch_{epoch+1:03d}.pth"),
                )
                print(f"🟢 Checkpoint saved for epoch {epoch+1}")

        if val_bal_acc > best_val_metric:
            best_val_metric = val_bal_acc
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            print("🟢 Valid" "ation Balanced Accuracy improved. Saving best model state...")

        else:
            epochs_without_improvement += 1
            print(f"🟡 No improvement for {epochs_without_improvement} epochs.")


        if epochs_without_improvement >= patience:
            print(f"🔴 Early stopping triggered at epoch {epoch+1}. Best Balanced Accuracy: {best_val_metric:.4f}")
            break

        print(f"Epoch [{epoch+1}/{n_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f} | adversarial F1 : {train_adv_f1:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f} | "
            f"Time: {time.time() - start_time:.1f}s | "
            f"Balanced Accuracy Val: {val_bal_acc:.4f} | - "
            f"Domain Loss: {domain_loss:.4f}")        

    writer.close()
    print("--- Training Finished ---")
    if best_model_state:
        print(f"Best validation Balanced Accuracy: {best_val_metric:.4f} achieved at Epoch {best_epoch}")
    return best_epoch





device = torch.device("cuda")

def predict(
        model,
        window_len=None,
        loader=None,
        num_samples_to_predict=None,
        num_classes=2,
        probability=False,
        device="cpu",
        ):
    

    """
    The function Performs prediction on EEG data using a sliding-window-based model architecture.
    
    This function is designed to be used with a custom EEGDataset (passed via a DataLoader),
    which already has preprocessed the EEG signals into temporal windows and provides:
    - `data`     : Tensor of shape (num_windows, channels, time)
    - `labels`   : Tensor of shape (num_windows,) — used only for consistency, not during inference
    - `weights`  : Tensor of shape (num_windows,) — soft importance values per window,
                  derived from the "Validation" EEG channel signal for use in aggregation

    The model outputs logits for each window, which are then grouped per trial (based on `window_len`)
    and ----softly averaged---- using the corresponding `weights`. This returns one prediction per original EEG trial.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model. Its forward method must return:
        - Class logits of shape [batch_size, num_classes]
        - Domain logits (ignored during prediction)

    window_len : int
        Length of a single window. Used to regroup flat predictions into per-trial decisions.

    loader : torch.utils.data.DataLoader
        DataLoader containing a dataset of type `EEGDataset`, which must expose the attributes:
        - `.data`, `.labels`, and `.weights`

    num_samples_to_predict : int
        Number of original EEG trials to predict. Used to preallocate output tensors.

    num_classes : int
        Number of classification classes (e.g., 2 for binary classification).

    probability : bool
        If True, returns softmax probabilities for each class; otherwise, returns discrete predictions.

    device : str
        Device for model inference ("cpu" or "cuda").

    Returns
    -------
    np.ndarray
        - If `probability=False`: array of shape (num_trials,) containing class predictions.
        - If `probability=True`: array of shape (num_trials, num_classes) with softmax probabilities.

    Notes
    -----
    Soft Averaging Strategy:
    ------------------------
    For each original trial, predictions from its `window counts` windows are softly averaged
    using the weights from the dataset's `Validation` channel. This reflects the confidence or quality
    of each window's data segment.

    The trial-level logits are computed as:

        logits_trial = sum(w_i * logits_i) / sum(w_i)

    where w_i is the validation-based weight for window i.

    Fallback Behavior:
    ------------------
    If all weights for a trial are close to zero (e.g., due to noisy validation signal),
    the function falls back to a **simple mean** of the window logits to avoid division by zero.

    Assumptions:
    ------------
    - The windows are arranged **sequentially per trial**, and `window_len` is consistent for all trials.
    - Domain predictions are not used during inference, so `domain_lambda` is set to 0 in the forward pass.
    """
    if isinstance(device,str):
        device = torch.device(device)

    model.eval()
    preds = []
    probs=[]
    with torch.no_grad():


        x = loader.dataset.data.detach().clone().to(device, dtype=torch.float32)
        y = loader.dataset.labels.detach().clone().to(device, dtype=torch.long) 
        weights = loader.dataset.weigths.detach().clone().to(device, dtype=torch.float32)
        outputs , _ = model(x,0)



        aggregated_outputs = torch.zeros((num_samples_to_predict, num_classes), device=device, dtype=torch.float32)

        k = 0 # Counter for original trials
        for i in range(0, outputs.shape[0], int(window_len)):
            logits = outputs[i : i + int(window_len)] 
            w_cur = weights[i : i + int(window_len)].unsqueeze(1) # [WINDOW_LEN, 1]

            # Re-enable weighted aggregation for validation!
            denom = w_cur.sum()
            if denom > 1e-8: # Add small epsilon to avoid division by zero
                aggregated_outputs[k] = (logits * w_cur).sum(dim=0) / denom
            else:
                # Fallback if weights are all zero for a trial's windows
                aggregated_outputs[k] = logits.mean(dim=0) # Use unweighted mean as fallback
            
            k += 1



        cpu_aggregated_outputs = aggregated_outputs.to("cpu")
        preds = torch.argmax(cpu_aggregated_outputs,axis = 1)

    if probability:
        return torch.softmax(cpu_aggregated_outputs, dim=1).numpy()
    else:
        return np.asarray(preds).reshape(-1,)