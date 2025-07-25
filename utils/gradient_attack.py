import torch

def GradientAttack(model,
                   x,
                   y,
                   loss_fn,
                   alpha=0.01,
                   epsilon=0.1,
                   steps=10,
                   clamp_min=-16.8080,
                   clamp_max=16.8080
                   ):
        """
    Performs a multi-step gradient-based adversarial attack (Projected-Gradient-Descent-style) on normalized inputs. Primarily for robust, model aware data augmentation.
    For Instance: Can you generate an augmented version of the inputs (x) That are strange to the model?

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to attack. Should be in `eval()` mode during attack.
    
    x : torch.Tensor
        Input tensor of shape (batch_size, n_channels, n_times), already standard normalized (z-scored).
    
    y : torch.Tensor
        Ground truth labels for x, of shape (batch_size,).
    
    alpha : float
        Step size for each gradient update.
    
    steps : int
        Number of gradient steps to perform.
    
    epsilon : float
        Maximum total L-infinity norm perturbation allowed (applied in normalized space).
    
    clamp_min : float
        Minimum value to clamp adversarial inputs (e.g., -3.0 for z-scored inputs).
    
    clamp_max : float
        Maximum value to clamp adversarial inputs (e.g., +3.0 for z-scored inputs).

    Returns:
    --------
    x_adv : torch.Tensor
        The adversarial version of `x` within the `epsilon` L∞-ball and clamped to valid range. In other words, A purturbed version of the inputs (x) which is designed to attack the model.

    Steps:
    ------
    1. Clone and detach the original input to avoid backpropagating through its history.
    2. Initialize x_adv with a clone (copy) of x, and set requires_grad=True so we can compute gradients w.r.t. it.
    3. For `steps` iterations:
        a. Forward pass: compute model predictions on x_adv.
        b. Compute the loss between predictions and true labels.
        c. Backward pass: compute gradients of the loss w.r.t. x_adv.
        d. Update x_adv by taking a step in the direction of the gradient (sign only).
        e. Project x_adv back into the L-infinity ε-ball centered at original x where: (x-epsilon)  <=  x_adv  <=  (x+epsilon)  .
        f. Clamp x_adv to ensure its values stay within the valid input range and ensure purturbed outputs (X_adv) distribution is similar to training data distribution.
        g. Re-enable gradient tracking on x_adv for the next iteration.
    4. Return the final adversarial example (x_adv) detached from the graph.
    """
        model.eval() #Freezes the model for attack generation (because I don't want to change model weights at this stage)
        x_adv = x.clone().detach().requires_grad_(True) #we set requires grad to True because we want to calculate gradients (for gradient attack generation)
        original = x.clone().detach()
        place_holder_lambda = 0
        for _ in range(steps):
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            logits,_ = model(x_adv,place_holder_lambda)
            adv_loss = loss_fn(logits,y)
            adv_loss.mean().backward()


            with torch.no_grad(): #Hey pytorch. I'm manually modifying this tensor. don't track it in your computational graph
                x_adv = x_adv + alpha*x_adv.grad.sign() #purturbation (Attack Generation)
                x_adv = torch.min(x_adv,original+epsilon) #clipping upper limit 
                x_adv = torch.max(x_adv,original-epsilon) #clipping lower limit

                x_adv = torch.clamp(x_adv, clamp_min, clamp_max)
            
            x_adv = x_adv.requires_grad_(True) #we re-enable gradient tracking here because torch.no_grad() disabled them

        return x_adv.detach() #detach because we don't need you're history (aka computational graph)

