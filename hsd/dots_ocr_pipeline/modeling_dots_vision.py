# Vendored from the dots.ocr model release (https://huggingface.co/rednote-hilab/dots.ocr),
# distributed under the MIT License by rednote-hilab (dots.ocr repo).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

flash_attn_available = True
npu_available = True

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_available = False

from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
from .configuration_dots import DotsVisionConfig

try:
    import torch_npu
except ImportError:
    npu_available = False


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()

    cos = freqs.cos()
    sin = freqs.sin()

    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()

    output = (tensor * cos) + (rotate_half(tensor) * sin)

    output = output.to(orig_dtype)

    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
            self,
            dim: int,
            context_dim: int,
            spatial_merge_size: int = 2,
            pre_norm="layernorm",
            init_merger_std=None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            print("no norm in patch merger")

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

        if init_merger_std is not None:
            nn.init.normal_(self.mlp[0].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[0].bias)
            nn.init.normal_(self.mlp[2].weight, mean=0.0, std=init_merger_std)
            nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class VisionAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config
        self.is_causal = config.is_causal

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )  # 'shd'
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=self.is_causal
        ).reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        return attn_output


class VisionAttentionV2(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        seqlens = torch.diff(cu_seqlens).tolist()

        q_list = torch.split(q, seqlens, 0)
        k_list = torch.split(k, seqlens, 0)
        v_list = torch.split(v, seqlens, 0)
        # eager attention is O(n^2) in memory (n = batch*seq); this variant splits
        # the sequence per batch to reduce memory at some speed cost.
        outputs = []
        for q_i, k_i, v_i in zip(q_list, k_list, v_list):
            q_i = q_i.transpose(0, 1)
            k_i = k_i.transpose(0, 1)
            v_i = v_i.transpose(0, 1)
            out = torch.matmul(q_i, k_i.transpose(1, 2)) / math.sqrt(self.head_dim)
            out = nn.functional.softmax(out, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(out, v_i)
            out = out.transpose(0, 1)
            outputs.append(out)

        attn_output = torch.concat(outputs, dim=0)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionAscendAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.ones([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = False

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_output = torch_npu.npu_prompt_flash_attention(q, k, v,
                                                           atten_mask=attention_mask,
                                                           num_heads=self.num_heads, input_layout="BNSD",
                                                           scale_value=self.head_dim ** -0.5)
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionSdpaAttention(nn.Module):
    def __init__(self, config, dim: int, num_heads: int = 16, bias=True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.config = config

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = True

        # Convert q, k, v to 4D to enable : (1, num_heads, seq_length, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)   # (1, num_heads, seq_length, head_dim)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # See: https://github.com/pytorch/pytorch/issues/127523
        if attention_mask.stride(-1) != 1:
            attention_mask = torch.empty_like(attention_mask, memory_format=torch.contiguous_format).copy_(attention_mask)

        # use memory efficient backend
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)

        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (seq_length, num_heads, head_dim)
        attn_output = attn_output.reshape(seq_length, -1)

        attn_output = self.proj(attn_output)
        return attn_output


DOTS_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "eager_v2": VisionAttentionV2,  # lower memory
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
    "ascend_fa": VisionAscendAttention,  # Ascend; accuracy degrades on long sequences
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class DotsPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        x = x.view(-1, self.num_channels, self.temporal_patch_size, self.patch_size, self.patch_size)[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        tokens = self.patchifier(x, grid_thw)
        return tokens


class DotsVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "flash_attention_2"):
        super().__init__()

        if attn_implementation == "flash_attention_2" and not flash_attn_available:
            # fallback to eager
            attn_implementation = "eager"
            print("flash attention not available! fallback to eager implementation ")

        if attn_implementation == "ascend_fa" and not npu_available:
            attn_implementation = "eager"
            print("flash attention not available! fallback to eager implementation ")

        self.attn = DOTS_VISION_ATTENTION_CLASSES[attn_implementation](
            config, config.embed_dim, num_heads=config.num_attention_heads, bias=config.use_bias
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):
    def __init__(self, config: DotsVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [DotsVisionBlock(config, config.attn_implementation) for _ in range(_num_hidden_layers)]
        )

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
        )

        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def get_pos_ids_by_grid(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            )

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, bf16=True) -> torch.Tensor:
        if bf16:
            hidden_states = hidden_states.bfloat16()
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb,
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states
