import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
from transformers import TrainingArguments, Trainer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

# hf datasets for easy oxford flowers training

from huggingface_hub import hf_hub_download
import scanpy as sc
import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import wandb
except ImportError:
    wandb = None

#path = hf_hub_download(repo_id='weililab/pancreatic_18clone', repo_type='dataset', filename='18clones_seurat.h5ad')
#ds = sc.read_h5ad(path)
data_path = Path(__file__).resolve().parent / "train_val.h5ad"
ds = sc.read_h5ad(data_path)
ds = ds.X.toarray()
#print(ds.shape)

class SCRNADataset(Dataset):
    def __init__(self, tokens, targets):
        self.tokens = tokens
        self.targets = targets

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # input_ids remain integers for the embedding layer
        input_tensor = torch.tensor(self.tokens[idx], dtype=torch.long)
        # labels become the raw continuous floats for MSE loss
        label_tensor = torch.tensor(self.targets[idx], dtype=torch.float)
        
        return {"input_ids": input_tensor, "labels": label_tensor}


class ValueEncoder(Module):
    def __init__(self, nbins, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(5053, embed_dim)
        
    def forward(self, x):
        return self.embedding(x)


class ExprDecoder(Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.ffn(x).squeeze(-1)


class Attention(Module):
    def __init__(
        self,
        dim,
        nheads = 4,
        dim_head = 32,
        flash=True,
    ):
        super().__init__()
        
        self.scale = dim_head ** -0.5
        self.nheads = nheads
        hidden_dim = dim_head * nheads
        
        self.flash = flash
        
        self.norm = torch.nn.modules.normalization.RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.nheads), qkv)

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
            attn = sim.softmax(dim=-1)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

        
class EncoderBlock(Module):
    def __init__(self, dim, nheads, dim_head, ff_mult=4):
        super().__init__()
        self.attn = Attention(dim, nheads, dim_head)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )
        """self.d_model = d_model
        #self.proj_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # (b, proj_qkv) -> (3, b, proj_qkv / 3)
        #self.attn = nn.MultiheadAttention(d_model, nheads, batch_first=True)
        #self.attn = Attention(d_model, nheads, d_model / nheads, flash=True)
        #self.ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model)
        )"""
    
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        """proj = self.proj_qkv(x)
        q, k, v = rearrange(proj, "b seq_len (n p) -> n b seq_len p", n=3, p=self.d_model)
        attn_output, _ = self.attn(q, k, v)
        attn_output = self.ln(attn_output)
        return self.ffn(attn_output)"""
        return x
      
        
class PerturbationModel(Module):
    def __init__(self, dim, nheads, dim_head, nlayers, nbins, output_dim, loss_fn):
        super().__init__()
        
        layers = []
        layers.append(ValueEncoder(nbins, d_model))
        for _ in range(nlayers):
            layers.append(EncoderBlock(d_model, nheads, dim_head))
        layers.append(ExprDecoder(d_model))
        self.net = nn.Sequential(*layers)
        
        self.loss_fn = loss_fn
        
    def forward(self, input_ids=None, labels=None, **kwargs):
        out = self.net(input_ids)
        loss = self.loss_fn(out, labels.float()) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=out)


def tokenizer(data, nbins):
    bins = np.linspace(data.min(), data.max(), nbins)
    binned = np.digitize(data, bins)
    
    return binned.astype(int)

# Create a new variable for the binned tokens
nbins = 100
tokens = tokenizer(ds, nbins)
print(tokens.dtype)

split = int(0.8 * len(ds))

# Split the binned tokens (Inputs)
train_tokens = tokens[:split]
test_tokens = tokens[split:]

# Split the raw continuous floats (Labels)
train_targets = ds[:split]
test_targets = ds[split:]

# Initialize datasets with both
training_data = SCRNADataset(train_tokens, train_targets)
test_data = SCRNADataset(test_tokens, test_targets)
d_model = 192
nheads = 12
dim_head = 16
num_layers = 6
output_dir = "pert-model-6"
num_train_epochs = 1
train_batch_size = 64
eval_batch_size = 64
gradient_accumulation_steps = 8
learning_rate = 2e-3
logging_steps = 50
wandb_project = "pertflow"
wandb_run_name = output_dir
wandb_mode = os.getenv("WANDB_MODE", "online")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader: DataLoader, model: Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    best_loss = 1000000
    count = 0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch, gene_expr in enumerate(dataloader):
            gene_expr = gene_expr.to(device)
            loss, pred = model(gene_expr)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(gene_expr)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
loss_fn = nn.MSELoss()
model = PerturbationModel(d_model, nheads, dim_head, num_layers, nbins, 1, loss_fn)
model = model.to(device)
torch.set_float32_matmul_precision('high')

wandb_run = None
if wandb is None:
    print("wandb not installed; continuing without wandb logging")
else:
    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        config={
            "d_model": d_model,
            "nheads": nheads,
            "num_layers": num_layers,
            "nbins": nbins,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": train_batch_size,
            "per_device_eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
        },
    )

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    remove_unused_columns=False,
    gradient_accumulation_steps=gradient_accumulation_steps,
    #gradient_checkpointing=True,
    bf16=True,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    report_to="wandb" if wandb_run is not None else "none",
    run_name=wandb_run_name,
    torch_compile=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=test_data,
    data_collator=default_data_collator,
)

#trainer.train()


print(model)
num_parameters = sum(p.numel() for p in model.parameters())
print(num_parameters)

if wandb_run is not None:
    wandb_run.summary["num_parameters"] = num_parameters
    wandb.finish()
    
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model.eval()
all_preds = []
all_targets = []

test_dataloader = DataLoader(test_data, batch_size=eval_batch_size)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids)
        all_preds.append(out.logits.float().cpu().numpy())
        all_targets.append(labels.float().cpu().numpy())

preds = np.concatenate(all_preds)
targets = np.concatenate(all_targets)

# Global metrics
print(f"MSE:  {mean_squared_error(targets, preds):.6f}")
print(f"MAE:  {mean_absolute_error(targets, preds):.6f}")
print(f"R²:   {r2_score(targets, preds):.6f}")

# Per-cell correlation (how well each cell's profile is reconstructed)
cell_pearson = [pearsonr(targets[i], preds[i])[0] for i in range(len(targets))]
print(f"\nPer-cell Pearson r:  {np.nanmean(cell_pearson):.4f} ± {np.nanstd(cell_pearson):.4f}")

# Per-gene correlation (how well each gene's variation across cells is captured)
gene_pearson = []
for g in range(targets.shape[1]):
    if targets[:, g].std() > 0:
        gene_pearson.append(pearsonr(targets[:, g], preds[:, g])[0])
print(f"Per-gene Pearson r:  {np.nanmean(gene_pearson):.4f} ± {np.nanstd(gene_pearson):.4f}")
print(f"  Genes with r > 0.6: {sum(1 for r in gene_pearson if r > 0.6)}/{len(gene_pearson)}")

# Fraction of variance unexplained per gene
gene_fvu = []
for g in range(targets.shape[1]):
    var = targets[:, g].var()
    if var > 0:
        gene_fvu.append(((targets[:, g] - preds[:, g])**2).mean() / var)
print(f"Per-gene FVU (median): {np.nanmedian(gene_fvu):.4f}")

#torch.save(model, "model_big4.pt")
"""
torch.set_float32_matmul_precision('high')
model = torch.load("model_big4.pt", map_location=device, weights_only=False)

pred_loss_fn = nn.L1Loss()
model.eval()
loss = 0
with torch.no_grad():
    for gene_expr in test_dataloader:
        gene_expr = gene_expr.to(device)
        
        batch_loss, pred = model(gene_expr)
        #batch_loss = pred_loss_fn(pred, gene_expr.to(torch.float32).squeeze())
        loss += batch_loss.item()

print(test_data.ds.shape)
#print(next(iter(test_dataloader)).shape)
print(loss / (test_data.ds.shape[0]) )
print(loss)



with torch.no_grad():
    sample_batch = next(iter(test_dataloader)).to(device)
    pred = model(sample_batch)
    print((pred - sample_batch).abs())
    #print(((pred - sample_batch).abs().sum(dim=1))/(100/(test_data.ds.max()-test_data.ds.min())))
    print(pred[0].shape)
    print(pred[0])
    print(sample_batch[0])
"""
"""
nano_flow = NanoFlow(
    model,
    predict_clean = True,
    times_cond_kwarg = 'times',
    normalize_data_fn = lambda t: t,
    unnormalize_data_fn = lambda t: t
)

trainer = Trainer(
    nano_flow,
    dataset = scrna_dataset,
    num_train_steps = 1_000,
    save_results_every = 1_000,
    checkpoint_every = 1_000,
    results_folder = './results'   # samples will be saved periodically to this folder
)

trainer()
"""
