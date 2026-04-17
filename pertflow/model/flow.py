from ema_pytorch import EMA
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.nn import Module
from einops import repeat
import math

from pertflow.utils.custom_utils import append_dims, exists, default, cosine_sim_loss, logit_normal_schedule

class RectifiedFlow(Module):
    """Simple rectified-flow objective on straight paths between source and target."""

    def __init__(
        self,
        model: Module,
        times_cond_kwarg="times",
        max_timesteps=100,
        loss_fn=F.mse_loss,
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.max_timesteps = max_timesteps
        self.loss_fn = loss_fn

    @torch.no_grad()
    def sample(self, source, cond, steps=16):
        """Integrate the learned vector field from a source state."""
        x = source.clone()
        delta = 1.0 / steps
        times = torch.linspace(0.0, 1.0, steps + 1, device=source.device)[:-1]

        for time in times:
            time_batch = repeat(time, "-> b", b=source.shape[0])
            time_kwarg = (
                {self.times_cond_kwarg: time_batch}
                if exists(self.times_cond_kwarg)
                else dict()
            )
            pred_flow = self.model(x, cond=cond, **time_kwarg)
            x = x + delta * pred_flow

        return x

    def forward(self, source, target, cond, times=None, loss_reduction="mean"):
        """Compute direct path flow loss at random interpolation times."""
        batch = source.shape[0]
        device = source.device

        if times is None:
            times = torch.rand(batch, device=device)
            times = times * (1.0 - self.max_timesteps**-1)

        padded_times = append_dims(times, source.ndim - 1)
        x_t = source.lerp(target, padded_times)
        velocity_target = target - source

        time_kwarg = (
            {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else dict()
        )
        pred_flow = self.model(x_t, cond=cond, **time_kwarg)
        return self.loss_fn(pred_flow, velocity_target, reduction=loss_reduction)
        

class SelfFlow(Module):
    """Source→target flow matching with an EMA teacher and representation alignment.

    Drop-in replacement for ``DirectPathFlow``: same ``(source, target, cond, times)``
    forward and ``(source, cond, steps)`` sample interface. Supervises the velocity
    ``target - source`` along the straight path ``source.lerp(target, t)`` while
    additionally aligning a student hidden state to an EMA-teacher hidden state
    computed at a cleaner (max-of-two) timestep, with per-patch dual-time masking
    on the student input.
    """

    def __init__(
        self,
        model: Module,
        teacher_model: Module | None = None,
        times_cond_kwarg="times",
        max_timesteps=100,
        loss_fn=F.mse_loss,
        repr_loss_fn=cosine_sim_loss,
        repr_loss_weight=1.0,
        mask_ratio=0.5,
        patch_size: int = 1,
        student_align_layer=-2,
        teacher_align_layer=-1,
        schedule_fn=logit_normal_schedule,
        eps=1e-5,
        ema_kwargs=None,
    ):
        super().__init__()
        self.model = model

        ema_kwargs = default(
            ema_kwargs,
            dict(beta=0.999, update_every=1, include_online_model=False),
        )
        self.teacher_model = default(teacher_model, EMA(model, **ema_kwargs))
        if not isinstance(self.teacher_model, EMA):
            raise TypeError("teacher model must be an instance of ema_pytorch.EMA")

        self.times_cond_kwarg = times_cond_kwarg
        self.max_timesteps = max_timesteps

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.schedule_fn = schedule_fn
        self.eps = eps

        self.student_align_layer = student_align_layer
        self.teacher_align_layer = teacher_align_layer

        dim = model.dim
        self.projector = nn.Sequential(
            nn.RMSNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

        self.has_repr_loss = repr_loss_weight > 0.0

        self.loss_fn = loss_fn
        self.repr_loss_fn = repr_loss_fn
        self.repr_loss_weight = repr_loss_weight

        self.register_buffer("zero", tensor(0.0), persistent=False)

    def post_training_step_update(self):
        """Update the EMA teacher after an optimizer step."""
        self.teacher_model.update()

    @torch.no_grad()
    def sample(self, source, cond, steps=16, model="teacher"):
        """Integrate the learned vector field from a source state."""
        if model not in ("student", "teacher"):
            raise ValueError(f"Unknown sample model {model!r}; expected 'student' or 'teacher'")
        if not 1 <= steps <= self.max_timesteps:
            raise ValueError(f"steps must be in [1, {self.max_timesteps}], got {steps}")

        selected_model = self.teacher_model if model == "teacher" else self.model

        x = source.clone()
        delta = 1.0 / steps
        times = torch.linspace(0.0, 1.0, steps + 1, device=source.device)[:-1]

        for t in times:
            t_batch = repeat(t, "-> b", b=source.shape[0])
            time_kwarg = (
                {self.times_cond_kwarg: t_batch}
                if exists(self.times_cond_kwarg)
                else dict()
            )
            out = selected_model(x, cond=cond, **time_kwarg)
            pred = out[0] if isinstance(out, tuple) else out
            x = x + delta * pred

        return x

    def forward(
        self,
        source,
        target,
        cond,
        times=None,
        loss_reduction="mean",
        return_loss_breakdown=False,
    ):
        """Compute the source→target SelfFlow objective and alignment loss."""
        shape, device = source.shape, source.device
        batch, seq_len = shape[0], shape[1]

        patch_size = self.patch_size
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if seq_len % patch_size != 0:
            raise ValueError(
                f"patch_size={patch_size} must evenly divide sequence length {seq_len}"
            )
        num_patches = seq_len // patch_size

        # Two time samples per example: explicit `times` override the student sample
        # so callers (and tests) can control the student interpolation exactly.
        if times is None:
            student_time = self.schedule_fn(torch.rand(batch, device=device))
        else:
            student_time = times
        teacher_time = self.schedule_fn(torch.rand(batch, device=device))

        # Teacher always sees a state at least as clean as the student's.
        times_clean_teacher = torch.maximum(teacher_time, student_time)

        flow = target - source

        teacher_time_patch = repeat(teacher_time, "b -> b n", n=num_patches)
        student_time_patch = repeat(student_time, "b -> b n", n=num_patches)

        mask = torch.rand((batch, num_patches), device=device) < self.mask_ratio
        times_clean_student_patch = torch.where(
            mask, student_time_patch, teacher_time_patch
        )

        times_student_elem = repeat(
            times_clean_student_patch, "b n -> b (n p)", p=patch_size
        )
        student_input = source.lerp(target, times_student_elem)

        time_kwarg = (
            {self.times_cond_kwarg: times_clean_student_patch}
            if exists(self.times_cond_kwarg)
            else dict()
        )

        pred, student_hiddens = self.model(
            student_input, cond=cond, return_hiddens=True, **time_kwarg
        )

        flow_loss = self.loss_fn(pred, flow, reduction=loss_reduction)

        repr_loss = self.zero
        if self.has_repr_loss:
            self.teacher_model.eval()
            with torch.no_grad():
                time_kwarg_teacher = (
                    {self.times_cond_kwarg: times_clean_teacher}
                    if exists(self.times_cond_kwarg)
                    else dict()
                )
                times_clean_teacher_padded = append_dims(
                    times_clean_teacher, source.ndim - 1
                )
                teacher_input = source.lerp(target, times_clean_teacher_padded)

                _, teacher_hiddens = self.teacher_model(
                    teacher_input, cond=cond, return_hiddens=True, **time_kwarg_teacher
                )

            student_repr = student_hiddens[self.student_align_layer]
            student_pred_teacher = self.projector(student_repr)
            teacher_repr = teacher_hiddens[self.teacher_align_layer]
            repr_loss = self.repr_loss_fn(student_pred_teacher, teacher_repr)

        total_loss = flow_loss + repr_loss * self.repr_loss_weight

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, flow_loss, repr_loss)


class SinusoidalPosEmb(Module):
    """Standard sinusoidal embedding for scalar diffusion or flow times."""

    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb  # (..., 1) * (half_dim,) -> (..., half_dim)
        # emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMBlock(Module):
    """Feed-forward block modulated by a time-dependent FiLM projection."""

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
    """Genewise velocity predictor conditioned on time and encoded context."""

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
        """Predict per-gene flow vectors, optionally returning hidden states."""
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
