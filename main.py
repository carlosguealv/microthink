import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
torch.set_float32_matmul_precision('medium')

# ------------------- Model Definition -------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expansion=2, dt_min=0.001, dt_max=0.1, d_ff=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expansion
        self.d_inner = d_model * expansion
        
        # Normalization and projections
        self.norm = nn.LayerNorm(d_model)
        
        # Input projection and gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        
        # Convolution for local context
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1),
            groups=self.d_inner,
            bias=True
        )
        
        # SSM parameters (S4D parametrization)
        # Log-space diagonal state matrix
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        
        # Input-dependent B and C projections
        self.B_proj = nn.Linear(self.d_inner, self.d_inner * d_state)
        self.C_proj = nn.Linear(self.d_inner, self.d_inner * d_state)
        
        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Delta projection (time-varying delta)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # MLP block
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff if d_ff is not None else d_model * 4)
        
    def forward(self, x):
        # Input shape: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        
        # Residual connection
        residual = x
        
        # Normalization
        x = self.norm(x)
        
        # Project input and split into x and gate
        x_and_gate = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x_proj, gate = x_and_gate.chunk(2, dim=-1)
        
        # Apply convolution for local context
        x_conv = self.conv(x_proj.transpose(1, 2))
        x_conv = x_conv[:, :, :seq_len].transpose(1, 2)
        
        # Generate delta (input-dependent step size)
        delta = torch.sigmoid(self.dt_proj(x_conv)) * (self.dt_max - self.dt_min) + self.dt_min
        
        # Generate SSM parameters
        # For A, we use a diagonal state matrix
        A = -torch.exp(self.A_log.unsqueeze(0).unsqueeze(0))  # [1, 1, d_inner, d_state]
        
        # Discretize continuous parameters (ZOH discretization)
        # Ā = exp(delta * A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # [batch, seq_len, d_inner, d_state]
        
        # Input-dependent B and C
        B = self.B_proj(x_conv).view(batch, seq_len, self.d_inner, self.d_state)
        C = self.C_proj(x_conv).view(batch, seq_len, self.d_inner, self.d_state)
        
        # B̄ = (Ā - I)A⁻¹B ≈ delta * B for small delta
        B_bar = delta.unsqueeze(-1) * B
        
        # Initialize state
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        # Sequential state update (for clarity - actual implementation uses parallel scan)
        ys = []
        for t in range(seq_len):
            # Update state: h_t = Āh_{t-1} + B̄x_t
            h = A_bar[:, t] * h + B_bar[:, t] * x_proj[:, t].unsqueeze(-1)
            
            # Generate output: y_t = Ch_t + Dx_t
            y = (C[:, t] * h).sum(dim=-1) + self.D * x_proj[:, t]
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # [batch, seq_len, d_inner]
        
        # Apply gate and SiLU activation
        y = y * F.silu(gate)
        
        # Output projection
        ssm_out = self.out_proj(y)
        
        # First residual connection
        x = residual + ssm_out
        
        # Second residual block (MLP)
        residual2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        
        return residual2 + x

class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_length, expansion=2, d_state=16, d_conv=4, d_ff=None):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expansion=expansion, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        B, L = x.shape
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.pos_emb[:, :L, :]
        x = token_embeddings + pos_embeddings
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x)

# ------------------- PyTorch Lightning Module -------------------
class MambaLightning(pl.LightningModule):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_length, expansion=2, d_state=16, d_conv=4, d_ff=None, lr=1e-4):
        super().__init__()
        self.model = MambaModel(vocab_size, d_model, num_layers, max_seq_length, expansion, d_state, d_conv, d_ff)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        inputs = input_ids[:, :-1]  # All but the last token
        targets = input_ids[:, 1:]  # All but the first token
        logits = self.model(inputs)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

# ------------------- Iterable Dataset Wrapper for Streaming -------------------
class StreamingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad token exists
    
    def tokenize_function(self, example):
        text = example["system_prompt"] + "\n" + example["question"] + "\n" + example["response"]
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        return tokenized["input_ids"].squeeze(0)  # Remove batch dimension
    
    def __iter__(self):
        for example in self.dataset:
            yield self.tokenize_function(example)

# ------------------- DataModule for Streaming Data -------------------
class OpenOrcaDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    
    def collate_fn(self, batch):
        """Pads sequences in a batch dynamically"""
        max_len = max(seq.size(0) for seq in batch)
        padded_batch = [
            torch.cat([seq, torch.full((max_len - seq.size(0),), self.tokenizer.pad_token_id)])
            for seq in batch
        ]
        return {"input_ids": torch.stack(padded_batch)}
    
    def get_dataset(self):
        return StreamingDataset(self.dataset, self.tokenizer)
    
    def train_dataloader(self):
        dataset_iter = self.get_dataset()
        return DataLoader(dataset_iter, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=2)

# ------------------- Training Script -------------------
def main():
    # Model hyperparameters
    vocab_size = GPT2TokenizerFast.from_pretrained("gpt2").vocab_size
    d_model = 256
    num_layers = 2
    max_seq_length = 512
    expansion = 2
    d_state = 16
    d_conv = 4
    d_ff = 1024
    batch_size = 4
    lr = 1e-3
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = MambaLightning(vocab_size, d_model, num_layers, max_seq_length, expansion, d_state, d_conv, d_ff, lr)
    datamodule = OpenOrcaDataModule(tokenizer, batch_size=batch_size)
    
    # Callbacks: Progress Bar & Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="mamba-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True
    )
    progress_bar = TQDMProgressBar(refresh_rate=1)
    
    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=5,
        callbacks=[checkpoint_callback, progress_bar],
        log_every_n_steps=10,
        limit_train_batches=100000,
    )
    
    # Resume from checkpoint if exists
    if os.path.exists("checkpoints/last.ckpt"):
        print(f"Resuming from checkpoint...")
        trainer.fit(model, datamodule=datamodule, ckpt_path="checkpoints/last.ckpt")
    else:
        trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
