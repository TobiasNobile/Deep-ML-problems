import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    optimizer.zero_grad()
    total_loss = 0
    for x, y in micro_batches:
        pred = model(x)
        loss = criterion(pred, y)
        loss /= len(micro_batches)
        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    return total_loss
