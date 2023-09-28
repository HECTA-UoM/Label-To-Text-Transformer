import torch, copy

class Trainer:

  def __init__(self, n_epochs:int, batch_size: int, checkpoint: int,
               vocab_size:int, pad_idx:int, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor):

    assert len(src_tokens)%batch_size==0, f"The number of training samples ({len(src_tokens)}) must be divisible by the Batch_size."

    self.checkpoint = checkpoint
    self.n_epochs = n_epochs

    self.vocab_size = vocab_size
    self.pad_idx = pad_idx

    self.src_tokens = torch.tensor(src_tokens)
    self.tgt_tokens = torch.tensor(tgt_tokens)

    self.batch_size = batch_size
    self.src_batches = torch.chunk(self.src_tokens, len(self.src_tokens)//self.batch_size, dim=0)
    self.tgt_batches = torch.chunk(self.tgt_tokens, len(self.tgt_tokens)//self.batch_size, dim=0)

  def train(self, model, criterion, optimizer):

    model.train()
    models, batch_passed = [], 0
    training_loss_logs = []

    for epoch in range(self.n_epochs):
        
        average_loss, loss_count = 0, 0
        
        for src_tokens_batch, tgt_tokens_batch in zip(self.src_batches, self.tgt_batches):
        
          optimizer.zero_grad()
          output = model(src_tokens_batch, tgt_tokens_batch[:, :-1])
          loss = criterion(output.contiguous().view(-1, self.vocab_size), tgt_tokens_batch[:, 1:].contiguous().view(-1))
          
          loss.backward()
          optimizer.step()
          batch_passed += 1
          print(f"Batch: {batch_passed*self.batch_size}. Batch Training Loss: {loss.item()}.") # Batch training loss

          # Save training loss for logs
          average_loss, loss_count = average_loss + loss.item(), loss_count + 1

          # Save model checkpoint
          if batch_passed%self.checkpoint == 0:
            models.append(copy.deepcopy(model)) # Try not to save deepcopy but state_dicts to save 5 times more memory
        
        # Save training loss logs
        training_loss_logs.append(round(average_loss/loss_count, 4))
        print(f"Epoch {epoch+1} passed.")
        
    return models, training_loss_logs
