import torch
from tqdm import trange
import numpy as np
def train_model(
        model,  
        optimizer, 
        train_loader, 
        val_loader, 
        loss_criterion, 
        validate_score, 
        tb_log, 
        num_epochs: int = 10, 
        model_save_path: str = "model.pth", 
        early_stop: bool = True, 
        save_model: bool = True):

    val_losses = []
    for epoch in trange(num_epochs, desc='epochs'):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Training loop
        for i, (inputs, labels, masks) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs).squeeze()
            loss = loss_criterion(outputs, labels, masks)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate the training loss

        avg_train_loss = running_loss / len(train_loader)
        tb_log.add_scalar('train/loss', avg_train_loss, epoch)

        # Evaluation loop
        val_score = 0.0
        with torch.no_grad():
            for inputs, labels, masks in val_loader:
                outputs = model(inputs).squeeze()
                score = validate_score(outputs.cpu(), labels.cpu(), masks.cpu())
                val_score += score

        val_score = val_score / len(val_loader)
        val_losses.append(val_score)
        tb_log.add_scalar('val/score', val_score, epoch)


        if early_stop and epoch > 20:
            print(val_score)
            if val_score < np.mean(val_losses[-5:]) and val_score > 0.03:  # todo
                print('early stop')
                if save_model:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': val_score
                    }, model_save_path)
                break
    return 