import math
import os
from pathlib import Path
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from ema_pytorch import EMA
import numpy as np
from einops import rearrange, repeat
from transformers import TrainingArguments, Trainer, default_data_collator
from transformers.modeling_outputs import SequenceClassifierOutput

from huggingface_hub import hf_hub_download
import scanpy as sc
import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import wandb

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def append_dims(t, dims):
    shape = t.shape
    ones = ((1,) * dims)
    return t.reshape(*shape, *ones)

def logit_normal_schedule(t, loc = 0.0, scale = 1.0):
    logits = torch.logit(t, eps = 1e-5)
    return 1. - torch.sigmoid(logits * scale + loc) # sticking with 0 -> 1 convention of noise to data

def cosine_sim_loss(x, y):
    return 1. - F.cosine_similarity(x, y, dim = -1).mean()
    
LossBreakdown = namedtuple('LossBreakdown', ['total', 'flow_loss', 'repr_loss'])

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

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb  # (..., 1) * (half_dim,) -> (..., half_dim)
        #emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMBlock(Module):
    def __init__(self, dim, time_dim, ff_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x, t):
        h = self.norm(x)
        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        # if t was (b, time_dim), scale/shift are (b, dim) — need unsqueeze
        # if t was (b, n, time_dim), scale/shift are (b, n, dim) — already matches h
        if scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        h = h * (1 + scale) + shift
        return x + self.ff(h)


class FlowHead(Module):
    def __init__(self, dim, depth=4, time_dim=None):
        super().__init__()
        self.dim = dim
        time_dim = default(time_dim, dim * 4)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.x_proj = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([FiLMBlock(dim, time_dim) for _ in range(depth)])
        self.out = nn.Linear(dim, 1)

    def forward(self, x, times, cond, return_hiddens=False):
        # x: (b, n_genes), cond: (b, n_genes, dim), times: (b,)
        h = self.x_proj(x.unsqueeze(-1)) + cond
        t = self.time_mlp(times)
        hiddens = []
        for block in self.blocks:
            h = block(h, t)
            if return_hiddens:
                hiddens.append(h)
        out = self.out(h).squeeze(-1)
        if return_hiddens:
            return out, hiddens
        return out


class SelfFlow(Module):
    """
    Chefer et al. from Black Forest Labs — adapted for 1D sequence data.
    """

    def __init__(
        self,
        model: Module,
        teacher_model: Module | None = None,
        times_cond_kwarg = 'times',
        data_shape: tuple[int, ...] | None = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        predict_clean = False,
        max_timesteps = 100,
        loss_fn = F.mse_loss,
        repr_loss_fn = cosine_sim_loss,
        repr_loss_weight = 1.0,
        mask_ratio = 0.5,
        patch_size: int = 1,
        student_align_layer = -2,
        teacher_align_layer = -1,
        schedule_fn = logit_normal_schedule,
        eps = 1e-5,
        ema_kwargs = dict(
            beta = 0.999,
            update_every = 1
        )
    ):
        super().__init__()
        self.model = model

        self.teacher_model = default(teacher_model, EMA(model, **ema_kwargs))
        assert isinstance(self.teacher_model, EMA), 'teacher model must be an instance of EMA'

        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.predict_clean = predict_clean
        self.max_timesteps = max_timesteps

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

        self.schedule_fn = schedule_fn

        self.eps = eps

        self.student_align_layer = student_align_layer
        self.teacher_align_layer = teacher_align_layer

        dim = model.dim
        self.projector = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.has_repr_loss = repr_loss_weight > 0.

        self.loss_fn = loss_fn
        self.repr_loss_fn = repr_loss_fn
        self.repr_loss_weight = repr_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def post_training_step_update(self):
        self.teacher_model.update()

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        model = None,
        **kwargs
    ):
        model = default(model, 'teacher')
        assert model in ('student', 'teacher')

        selected_model = self.teacher_model if model == 'teacher' else self.model

        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'

        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)
        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]

        delta_time = 1. / steps
        x = noise

        for t in times:
            t = repeat(t, '-> b', b = batch_size)
            padded_t = append_dims(t, x.ndim - 1)
            time_kwarg = {self.times_cond_kwarg: t} if exists(self.times_cond_kwarg) else dict()

            model_output = selected_model(x, **time_kwarg, **kwargs)
            pred = model_output[0] if isinstance(model_output, tuple) else model_output

            if self.predict_clean:
                pred = (pred - x) / (1. - padded_t).clamp(min = self.eps)

            x = x + delta_time * pred

        out = self.unnormalize_data_fn(x)
        return out if not return_noise else (out, noise)

    def forward(
        self,
        data,
        noise = None,
        return_loss_breakdown = False,
        **kwargs
    ):
        shape, ndim, device = data.shape, data.ndim, data.device
        self.data_shape = default(self.data_shape, shape[1:])

        batch, seq_len = shape[0], shape[1]
        patch_size = self.patch_size
        num_patches = seq_len // patch_size

        # normalize

        data = self.normalize_data_fn(data)

        # times

        teacher_time = self.schedule_fn(torch.rand(batch, device = device))
        student_time = self.schedule_fn(torch.rand(batch, device = device))

        times_clean_teacher = torch.maximum(teacher_time, student_time)

        # noise and derive flow

        noise = default(noise, torch.randn_like(data))
        flow = data - noise

        # Dual-Timestep Scheduling — per-patch times along the 1D sequence

        teacher_time_patch = repeat(teacher_time, 'b -> b n', n = num_patches)
        student_time_patch = repeat(student_time, 'b -> b n', n = num_patches)

        # random mask for student

        mask = torch.rand((batch, num_patches), device = device) < self.mask_ratio

        times_clean_student_patch = torch.where(mask, student_time_patch, teacher_time_patch)

        # times for the teacher

        if self.predict_clean:
            times_clean_teacher = times_clean_teacher * (1. - self.max_timesteps ** -1)

        # upsample patch times to element-wise for lerp

        times_student_elem = repeat(times_clean_student_patch, 'b n -> b (n p)', p = patch_size)

        student_input = noise.lerp(data, times_student_elem)

        # pass per-patch times to student model

        time_kwarg = {self.times_cond_kwarg: times_clean_student_patch} if exists(self.times_cond_kwarg) else dict()

        pred, student_hiddens = self.model(
            student_input,
            return_hiddens = True,
            **time_kwarg,
            **kwargs
        )

        if self.predict_clean:
            pred = (pred - student_input) / (1. - times_student_elem).clamp(min = self.eps)

        # main loss

        flow_loss = self.loss_fn(flow, pred)

        # representation alignment loss

        repr_loss = self.zero

        if self.has_repr_loss:
            self.teacher_model.eval()

            with torch.no_grad():

                time_kwarg_teacher = {self.times_cond_kwarg: times_clean_teacher} if exists(self.times_cond_kwarg) else dict()

                times_clean_teacher_padded = append_dims(times_clean_teacher, data.ndim - 1)

                teacher_input = noise.lerp(data, times_clean_teacher_padded)

                _, teacher_hiddens = self.teacher_model(
                    teacher_input,
                    return_hiddens = True,
                    **time_kwarg_teacher,
                    **kwargs
                )

            student_repr = student_hiddens[self.student_align_layer]
            student_pred_teacher = self.projector(student_repr)

            teacher_repr = teacher_hiddens[self.teacher_align_layer]

            repr_loss = self.repr_loss_fn(student_pred_teacher, teacher_repr)

        total_loss = (
            flow_loss +
            repr_loss * self.repr_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, flow_loss, repr_loss)


class NanoFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = None,
        data_shape = None,
        normalize_data_fn = identity,
        unnormalize_data_fn = identity,
        predict_clean = False,
        max_timesteps = 100,
        loss_fn = F.mse_loss
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape

        self.normalize_data_fn = normalize_data_fn
        self.unnormalize_data_fn = unnormalize_data_fn

        self.predict_clean = predict_clean # predicting x0
        self.max_timesteps = max_timesteps

        self.loss_fn = loss_fn

    @torch.no_grad()
    def sample(
        self,
        steps = 16,
        batch_size = 1,
        data_shape = None,
        return_noise = False,
        **kwargs
    ):
        assert 1 <= steps <= self.max_timesteps

        data_shape = default(data_shape, self.data_shape)
        assert exists(data_shape), 'shape of the data must be passed in, or set at init or during training'
        device = next(self.model.parameters()).device

        noise = torch.randn((batch_size, *data_shape), device = device)

        times = torch.linspace(0., 1., steps + 1, device = device)[:-1]
        delta = 1. / steps

        denoised = noise

        for time in times:
            time = time.expand(batch_size)
            time_kwarg = {self.times_cond_kwarg: time} if exists(self.times_cond_kwarg) else dict()

            model_output = self.model(denoised, **time_kwarg, **kwargs)

            if self.predict_clean:
                padded_time = append_dims(time, denoised.ndim - 1)
                pred_flow = (model_output - denoised) / (1. - padded_time)
            else:
                pred_flow = model_output

            denoised = denoised + delta * pred_flow

        out = self.unnormalize_data_fn(denoised)

        if not return_noise:
            return out

        return out, noise

    def forward(self, data, noise = None, times = None, loss_reduction = 'mean', **kwargs):
        data = self.normalize_data_fn(data)

        # shapes and variables

        shape, ndim = data.shape, data.ndim
        self.data_shape = default(self.data_shape, shape[1:]) # store last data shape for inference
        batch, device = shape[0], data.device

        # flow logic

        times = default(times, torch.rand(batch, device = device))
        times = times * (1. - self.max_timesteps ** -1)

        noise = default(noise, torch.randn_like(data))
        flow = data - noise # flow is the velocity from noise to data, also what the model is trained to predict

        padded_times = append_dims(times, ndim - 1)
        noised_data = noise.lerp(data, padded_times) # noise the data with random amounts of noise (time) - lerp is read as noise -> data from 0. to 1.

        time_kwarg = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict() # maybe time conditioning, could work without it (https://arxiv.org/abs/2502.13129v1)
        model_output = self.model(noised_data, **time_kwarg, **kwargs)

        if self.predict_clean:
            pred_flow = (model_output - noised_data) / (1. - padded_times)
        else:
            pred_flow = model_output

        return self.loss_fn(flow, pred_flow, reduction = loss_reduction)

class PerturbationModel(Module):
    def __init__(self, dim, nheads, dim_head, nlayers, nbins, head_type="flow"):
        super().__init__()
        self.head_type = head_type
        self.encoder = nn.Sequential(
            ValueEncoder(nbins, dim),
            *[EncoderBlock(dim, nheads, dim_head) for _ in range(nlayers)],
        )
        if head_type == "flow":
            # include_online_model=False so EMA keeps a non-registered reference
            # to the online model; otherwise its parameters alias `self.head.model.*`
            # and safetensors refuses to serialize the duplicates on checkpoint save.
            self.head = SelfFlow(
                FlowHead(dim),
                times_cond_kwarg="times",
                ema_kwargs=dict(beta=0.999, update_every=1, include_online_model=False),
            )
        elif head_type == "mlp":
            self.head = ExprDecoder(dim)
        else:
            raise ValueError(f"unknown head_type: {head_type!r} (expected 'flow' or 'mlp')")

    def forward(self, input_ids=None, labels=None, **kwargs):
        cond = self.encoder(input_ids)  # (b, n_genes, d)

        if self.head_type == "mlp":
            preds = self.head(cond)  # (b, n_genes)
            loss = F.mse_loss(preds, labels.float()) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=preds)

        # flow head
        if self.training:
            loss = self.head(labels.float(), cond=cond)
            self.head.post_training_step_update()
            return SequenceClassifierOutput(loss=loss, logits=None)

        # eval / inference path — sample real predictions
        sampled = self.head.sample(
            batch_size=cond.shape[0],
            data_shape=(cond.shape[1],),
            cond=cond,
        )
        loss = F.mse_loss(sampled, labels.float()) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=sampled)


def tokenizer(data, nbins):
    bins = np.linspace(data.min(), data.max(), nbins)
    binned = np.digitize(data, bins)

    return binned.astype(int)

if __name__ == "__main__":
    #path = hf_hub_download(repo_id='weililab/pancreatic_18clone', repo_type='dataset', filename='18clones_seurat.h5ad')
    #ds = sc.read_h5ad(path)
    data_path = Path(__file__).resolve().parent / "train_val.h5ad"
    ds = sc.read_h5ad(data_path)
    ds = ds.X.toarray()

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
    # Ablation switch: "flow" for flow-matching head, "mlp" for ExprDecoder
    # baseline. Env var `HEAD_TYPE` overrides the default so you can run both
    # without editing the file: `HEAD_TYPE=mlp uv run python pertflow.py`.
    head_type = os.getenv("HEAD_TYPE", "flow")
    assert head_type in ("flow", "mlp"), f"HEAD_TYPE must be 'flow' or 'mlp', got {head_type!r}"

    d_model = 128
    nheads = 8
    dim_head = 16
    num_layers = 3
    output_dir = f"pert-model-self-{head_type}"
    num_train_epochs = 1
    train_batch_size = 64
    eval_batch_size = 64
    gradient_accumulation_steps = 8
    learning_rate = 1e-3
    logging_steps = 50
    wandb_project = "pertflow"
    wandb_run_name = output_dir
    wandb_mode = os.getenv("WANDB_MODE", "online")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = PerturbationModel(d_model, nheads, dim_head, num_layers, nbins, head_type=head_type)
    model = model.to(device)
    torch.set_float32_matmul_precision('high')


    wandb_run = wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        mode=wandb_mode,
        config={
            "head_type": head_type,
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
        torch_compile=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=test_data,
        data_collator=default_data_collator,
    )

    trainer.train()

    print(model)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(num_parameters)

    wandb_run.summary["num_parameters"] = num_parameters
    wandb.finish()

    # Save the arch config next to the Trainer's safetensors checkpoints so
    # eval.py can reconstruct `PerturbationModel` without hard-coding values.
    import json

    config_path = Path(output_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(
            {
                "head_type": head_type,
                "d_model": d_model,
                "nheads": nheads,
                "dim_head": dim_head,
                "num_layers": num_layers,
                "nbins": nbins,
            },
            f,
            indent=2,
        )
    print(f"Saved model config to {config_path}")
