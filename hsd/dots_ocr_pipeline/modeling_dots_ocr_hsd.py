"""
HSD modeling for dots.ocr: `DotsOCRForCausalLM` extended with
`batch_generate`, a speculative-decoding generation loop that verifies
pipeline drafts (PP-StructureV3 region outputs) against the model.

Core ideas (see the paper for details):
- Stage 1: all region drafts are packed into one batch (block-diagonal causal
  attention) and verified in parallel; accepted draft tokens are emitted for
  free instead of being decoded one by one.
- Stage 2: the full-page input is appended as the last block and verifies the
  refined outputs to preserve page-level coherence.
- Draft alignment uses KMP matching over token sequences; unmatched suffixes
  are proposed as speculation trees (prefix-tree batching).
- Attention runs on torch.compile'd FlexAttention with a 128x64 BlockMask
  grid; the KV cache is periodically compacted (HSD_COMPACT_PERIOD).

Tuning knobs are read from HSD_* environment variables (see README).
"""

from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from torch.nn.attention.flex_attention import BlockMask
import transformers.integrations.flex_attention as flex_attn_mod
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from .configuration_dots import DotsVisionConfig, DotsOCRConfig
from .modeling_dots_vision import DotsVisionTransformer
from copy import deepcopy

from collections import deque
import re
import time
import unicodedata
import string
import json
import codecs
import ast
import ijson
import io
import math

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention)

import os
_HSD_PROFILE = os.environ.get("HSD_PROFILE", "0") == "1"
_HSD_VERBOSE = os.environ.get("HSD_VERBOSE", "0") == "1"
_HSD_TIGHT_PACK = os.environ.get("HSD_TIGHT_PACK", "0") == "1"
_HSD_Q_BLOCK = 128  # pad Q to multiples of this (must match flex_attention warmup)
# HSD_COMPACT_PERIOD: physical KV compaction every N iters (1=every iter, 3=default).
# On non-compact iters we use cheap masked_fill(-inf) so BlockMask skips dead chunks.
# Trades slightly larger KV (negligible) for ~(N-1)/N fewer index_select kernel calls.
_HSD_COMPACT_PERIOD = int(os.environ.get("HSD_COMPACT_PERIOD", "3"))
# Max spec proposal tokens per block (0=unlimited). Caps draft length to avoid
# wasting forward compute on tokens that won't be accepted (typical acceptance
# position is ~16, so 32-64 captures most useful tokens).
_HSD_MAX_SPEC_LEN = int(os.environ.get("HSD_MAX_SPEC_LEN", "0"))
_HSD_MAX_TREE_LEN = int(os.environ.get("HSD_MAX_TREE_LEN", "0"))  # override max_tree_len (0=use default)

def _hsd_sync():
    if _HSD_PROFILE:
        torch.cuda.synchronize()

_CACHED_BLOCK_MASK_SRC = None
_CACHED_BLOCK_MASK = None


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    elif attention_mask is not None:
        # Raw 4D float mask (e.g. passed through transformers>=4.53
        # `create_causal_mask`, which returns 4D tensors as-is). Convert it
        # to a BlockMask here — previously this branch silently dropped the
        # mask and ran bidirectional attention.
        # The same mask tensor is shared by ALL layers within one forward:
        # cache the conversion so create_block_mask runs once per step, not
        # once per layer (28x per-step overhead otherwise).
        global _CACHED_BLOCK_MASK_SRC, _CACHED_BLOCK_MASK
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        if _CACHED_BLOCK_MASK_SRC is not attention_mask or _CACHED_BLOCK_MASK is None:
            from torch.nn.attention.flex_attention import create_block_mask
            _mask_mod = make_mask_mod(causal_mask, 128, 64)
            _CACHED_BLOCK_MASK = create_block_mask(
                _mask_mod, None, None,
                causal_mask.shape[-2], causal_mask.shape[-1],
                device=causal_mask.device,
                BLOCK_SIZE=(128, 64),
            )
            _CACHED_BLOCK_MASK_SRC = attention_mask
        block_mask = _CACHED_BLOCK_MASK

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # When running TP this helps:
    if not ((num_local_query_heads & (num_local_query_heads - 1)) == 0):
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options", None)
    attn_output, attention_weights = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def make_mask_mod(attention_mask: torch.Tensor, Q_BLOCK_SIZE: int, KV_BLOCK_SIZE: int):
    # attention_mask: [B, H, Q, KV], float, on GPU
    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        return attention_mask[0][0][q_idx][kv_idx] > -0.1

    return mask_mod

def to_halfwidth_clean(s: str) -> str:
    """
    NFKC-normalize full-width chars to half-width and drop
    newlines/carriage returns/spaces.
    """
    if s is None:
        return s

    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\n\n', '_SAVE_TAG_')
    s = s.replace('\n', '').replace('\r', '')
    s = s.replace('_SAVE_TAG_', '\n\n')

    s = s.replace('\\n\\n', '_SAVE_TAG_')
    s = s.replace('\\n', '').replace('\\r', '')
    s = s.replace('_SAVE_TAG_', '\\n\\n')

    s = s.replace("  ", "")
    return s

# 2. characters that full-width -> half-width conversion handles
chars = ["　"] + [chr(i) for i in range(0xFF01, 0xFF5F)] + ["  ", "   ", "    ", " <", "\n", "\r", ">\n"]  # + ["\\n", "\\r"]

def clean_string_and_get_maps(input_string):
    """
    Strip all symbols/whitespace, keeping CJK, kana, letters and digits,
    and return index maps between the original and cleaned strings.

    Returns a dict with 'cleaned_string', 'new_to_old_map' (cleaned idx ->
    original idx) and 'old_to_new_map' (original idx -> cleaned idx).
    """
    # character classes to keep
    # (precompiled for speed)
    keep_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ffa-zA-Z0-9]')
    
    cleaned_chars = []
    new_to_old_map = []
    old_to_new_map = {}
    
    new_idx_counter = 0
    # enumerate for original indices
    for old_idx, char in enumerate(input_string):
        # keep this character class?
        if keep_pattern.match(char):
            # keep
            cleaned_chars.append(char)
            
            # record mapping
            # new idx -> original idx
            new_to_old_map.append(old_idx)
            # original idx -> new idx
            old_to_new_map[old_idx] = new_idx_counter
            
            # advance the cleaned index
            new_idx_counter += 1
            
    # join kept characters
    cleaned_string = "".join(cleaned_chars)
    
    return {
        'cleaned_string': cleaned_string,
        'new_to_old_map': new_to_old_map,
        'old_to_new_map': old_to_new_map
    }

def unescape_string(s):
  """
  Unescape JSON-style escape sequences, e.g. 'Hello\\nWorld' -> 'Hello\nWorld'.
  
    
  """
  # 'unicode_escape' handles Python-literal-style escapes; encode to bytes
  # first (latin-1 maps each char 0-255 to one byte).
  return codecs.decode(s.encode('latin-1'), 'unicode_escape')

def unescape_string_robust(s: str) -> str:
    try:
        return codecs.decode(s.encode('latin-1'), 'unicode_escape')
    except UnicodeEncodeError:
        return s

def unescape_string_final(s: str) -> str:
    """
    Unescape via ast.literal_eval; handles strings mixing Unicode (CJK)
    and escape sequences (e.g. \\n) correctly.

      
    """
    try:
        # Wrap in quotes for ast.literal_eval, escaping inner double quotes.
        # quotes inside strings are rare for LLM output
        # simpler: wrap with explicit quotes
        # so use a custom wrapper
        
        # wrap in a single/double-quoted string
        s_escaped = s.replace('"', '\\"')  # escape inner double quotes
        s_quoted = '"' + s_escaped + '"'
        
        return ast.literal_eval(s_quoted)
        
    except (ValueError, SyntaxError):
        # fall back to the raw string if literal_eval fails
        return s

_CLOSING_QUOTE_RE = re.compile(r'(?<!\\)"')  # unescaped " for text/category close

class StreamingTextWatcher:
    """
    Efficient streaming JSON field tracker using string rfind instead of ijson.

    Old approach: rebuild a full ijson parser from byte-0 on every feed() call.
    Cost: O(n * ijson_overhead) per call → O(n²) total over N tokens.

    New approach: accumulate decoded text, use rfind to locate the most recent
    field marker, then check for the closing delimiter in the suffix.
    Cost: O(n) per call, much smaller constant.

    Limitation: naive bare-quote check for text/category may misfire if the
    JSON-encoded text value itself contains a literal " character (e.g. from
    escaped quotes \\"). We use a lookbehind regex to handle this correctly.
    """
    _BBOX_MARKER = '"bbox": ['
    _TEXT_MARKER = '"text": "'
    _CAT_MARKER  = '"category": "'

    def __init__(self):
        self._buf = ""
        self.text_mode = False
        self.bbox_mode = False
        self.category_mode = False
        self.element_count = 0       # number of completed elements (closed "}")
        self._last_element_count = 0

    def feed(self, chunk: bytes):
        self._buf += chunk.decode('utf-8', errors='replace')
        s = self._buf

        # Track element count: each "}" closes one JSON element
        self.element_count = s.count('"bbox": [')
        # bbox_entering: True on the iteration when bbox_mode first becomes True for this element
        prev_bbox = self.bbox_mode

        # bbox: look for most recent '"bbox": [', check if ']' has appeared after
        bi = s.rfind(self._BBOX_MARKER)
        self.bbox_mode = (bi != -1) and (']' not in s[bi + len(self._BBOX_MARKER):])

        # text: look for most recent '"text": "', check for unescaped closing "
        ti = s.rfind(self._TEXT_MARKER)
        if ti == -1:
            self.text_mode = False
        else:
            after_text = s[ti + len(self._TEXT_MARKER):]
            self.text_mode = _CLOSING_QUOTE_RE.search(after_text) is None

        # category: look for most recent '"category": "', check for unescaped closing "
        ci = s.rfind(self._CAT_MARKER)
        if ci == -1:
            self.category_mode = False
        else:
            after_cat = s[ci + len(self._CAT_MARKER):]
            self.category_mode = _CLOSING_QUOTE_RE.search(after_cat) is None

############################### patched source ###############################
import transformers.models.qwen2.modeling_qwen2 as org_modeling_qwen2

def new_init(self, config: org_modeling_qwen2.Qwen2Config, layer_idx: int):
    super(org_modeling_qwen2.Qwen2Attention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    

org_modeling_qwen2.Qwen2Attention.__init__ = new_init

def attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
):    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = org_modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    # default to eager attention
    attention_interface: org_modeling_qwen2.Callable = org_modeling_qwen2.eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            org_modeling_qwen2.logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            if attention_mask is not None and len(attention_mask.shape) == 4:
                attention_interface = flex_attention_forward
            else:
                attention_interface = org_modeling_qwen2.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

org_modeling_qwen2.Qwen2Attention.forward = attention_forward

def keep_flex_mask_update_causal_mask(self,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool = False,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and past_key_values is not None:
            is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    # flex_attention: convert mask below / return as-is
    if self.config._attn_implementation == "flex_attention":
        if isinstance(attention_mask, torch.Tensor):
            Q_LEN = attention_mask.shape[-2]
            KV_LEN = attention_mask.shape[-1]
            # NOTE: the previous implementation built the BlockMask via
            # _create_sparse_block_from_block_mask, which produced an incorrect
            # block layout on newer torch (2x more blocks than needed, wrong
            # attention). create_block_mask with a mask_mod closure is the
            # supported API and produces exact results (verified vs eager).
            _mask_mod = make_mask_mod(attention_mask, 128, 64)
            from torch.nn.attention.flex_attention import create_block_mask
            sparse_block_mask = create_block_mask(
                _mask_mod, None, None, Q_LEN, KV_LEN, device=attention_mask.device,
                BLOCK_SIZE=(128, 64),
            )

            return sparse_block_mask
    # sdpa/eager: return the mask as-is
    if self.config._attn_implementation == "sdpa" or self.config._attn_implementation == "eager":
        if isinstance(attention_mask, torch.Tensor):
            return attention_mask

    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, org_modeling_qwen2.StaticCache)
    using_sliding_window_cache = isinstance(past_key_values, org_modeling_qwen2.SlidingWindowCache)

    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if (
        self.config._attn_implementation == "sdpa"
        and not (using_static_cache or using_sliding_window_cache)
        and not output_attentions
    ):
        if org_modeling_qwen2.AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            sliding_window=self.config.sliding_window,
            is_training=self.training,
        ):
            return None

    dtype = input_tensor.dtype
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    # SlidingWindowCache or StaticCache
    if using_sliding_window_cache or using_static_cache:
        target_length = past_key_values.get_max_cache_shape()
    # DynamicCache or no cache
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
        config=self.config,
        past_key_values=past_key_values,
    )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type in ["cuda", "xpu", "npu"]
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = org_modeling_qwen2.AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

    return causal_mask

org_modeling_qwen2.Qwen2Model._update_causal_mask = keep_flex_mask_update_causal_mask

############################### patched source ###############################
from transformers.models.qwen2 import Qwen2ForCausalLM


############################### helper functions ###############################
class Node:
    def __init__(self, value, seg_id, token_pos, path, depth=0):
        self.value = value      # node value
        self.depth = depth      # node depth
        self.seg_id = seg_id  # segment index of this node
        self.token_pos = token_pos  # position within the segment
        self.path = path
        self.input_ids_index = -1
        self.children = {}      # child nodes keyed by token id

empty_node = Node(-1, -1, -1, (), -1)

def dfs(node):
    if node is None:
        return
    print(f"{node.depth}:{node.value}", end=" ")
    for child in node.children.values():
        dfs(child)

def bfs(root, tokenizer=None):
    if root is None:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if len(node.children) == 0:
            print([tokenizer.decode(i) for i in node.path[1:]], flush=True)  # current node path
        # iterate over child nodes
        for child in node.children.values():
            queue.append(child)

# compute the KMP failure table
def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0  # current longest prefix-suffix length
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

# KMP: find all match positions
def kmp_search(item_idx, text, pattern, lps, one_matches_tree, one_match_candidate_len):
    text_len = len(text)
    pattern_len = len(pattern)

    exact_match = []
    i = j = 0  # i: text idx, j: pattern idx
    pat_last_ch = pattern[-1]

    while i < text_len:
        text_i = text[i]

        # check pattern[-1] every time
        if text_i == pat_last_ch:
            # skip end-of-pattern-only matches
            now_node = one_matches_tree
            remaining_len = min(text_len - i - 1,
                                one_match_candidate_len)
            k = 0
            while k < remaining_len:
                k += 1  # true depth
                i_add_k = i + k
                now_node.children[text[i_add_k]] = \
                    now_node.children.get(text[i_add_k],
                                          Node(text[i_add_k], item_idx, i_add_k,
                                               now_node.path + (text[i_add_k], ), k))
                now_node = now_node.children[text[i_add_k]]

        if text_i == pattern[j]:
            i += 1
            j += 1
            if j == pattern_len:
                # skip matches that only touch the tail
                start = i - j
                # (>= would decode one extra token)
                if start + pattern_len <= text_len - 1:
                    exact_match.append((item_idx, start, text_len))
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return one_matches_tree, exact_match

def find_all_matches_kmp(x, q_list, lps, one_match_tree_depth):
    # -1 offset simplifies slicing later
    one_matches_tree = Node(q_list[-1], -1, -1, (-1, q_list[-1],), 0)
    exact_matches = []  # (item, start, end)
    for i, item in enumerate(x):
        (one_matches_tree, exact_match) = kmp_search(
            i, item, q_list, lps,
            one_matches_tree, one_match_tree_depth
        )
        if exact_match:
            exact_matches.extend(exact_match)

    return one_matches_tree, exact_matches

import transformers
# Use path relative to this file so it works from any working directory.
# The tokenizer files live next to the model weights (weights/dots.ocr); resolve
# them lazily so importing this module does not require the weights to be present.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_pre_tokenizer = None


def _get_pre_tokenizer():
    global _pre_tokenizer
    if _pre_tokenizer is None:
        import os as _os
        candidates = [
            _os.environ.get("HSD_TOKENIZER_PATH", ""),
            _THIS_DIR,  # tokenizer files vendored next to this module
            "./weights/dots.ocr",
        ]
        for c in candidates:
            if c and _os.path.isfile(_os.path.join(c, "tokenizer_config.json")):
                _pre_tokenizer = transformers.AutoTokenizer.from_pretrained(c, use_fast=True)
                break
        else:
            raise RuntimeError(
                "Could not locate the dots.ocr tokenizer. Set HSD_TOKENIZER_PATH "
                "or run from a directory containing weights/dots.ocr."
            )
    return _pre_tokenizer
# select tree also covers category candidates
def get_select_mode_one_matches_tree():
    begin = _get_pre_tokenizer().encode("[")[0]
    all_draft = [
        '{"bbox": [',              # after [ or after },
        ' "text": "',              # after ], or ",
        ', {"bbox": [',            # between elements: "} → , {"bbox": [
        '}]',                       # end of JSON array
    ]
    all_categories = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
    all_draft += [f' "category": "{cat}' for cat in all_categories]
    # Also add full field transitions for better tree depth
    for cat in all_categories:
        all_draft.append(f' "category": "{cat}", "text": "')

    tokenzied_draft = _get_pre_tokenizer()(all_draft, add_special_tokens=False)['input_ids']
    tokenzied_draft = [[begin,] + ids for ids in tokenzied_draft]  # prepend the begin node

    one_matches_tree, _ = find_all_matches_kmp(
        tokenzied_draft,
        [begin,],
        compute_lps([begin,]),
        15,  # deeper tree for longer transitions
    )
    return one_matches_tree

CACHED_SELECT_MODE_ONE_MATCHES_TREE = None

def get_select_mode_candidate(begin):
    '''
    Candidate nodes for select mode
    '''
    global CACHED_SELECT_MODE_ONE_MATCHES_TREE
    if CACHED_SELECT_MODE_ONE_MATCHES_TREE is None:
        CACHED_SELECT_MODE_ONE_MATCHES_TREE = get_select_mode_one_matches_tree()
    CACHED_SELECT_MODE_ONE_MATCHES_TREE.value = begin
    
    return CACHED_SELECT_MODE_ONE_MATCHES_TREE


# alternative construction
def get_bbox_mode_one_matches_tree():
    """Build a deep tree for bbox digit-by-digit decoding.

    Tokenizer encodes each digit as a single token (ids 15-24 for '0'-'9').
    Bbox format: [d+, d+, d+, d+] where d+ is 1-4 digits.

    We build the tree directly (not via KMP) to get proper depth:
    - digit node → children: 10 digits + comma + close_bracket
    - comma node → children: space
    - space node → children: 10 digits
    This covers full bbox sequences like "1243, 114, 2591, 255]" = 21 tokens.
    """
    begin = _get_pre_tokenizer().encode(" [")[0]
    # Token IDs for digits 0-9, comma, space, close bracket
    digit_ids = [_get_pre_tokenizer().encode(str(d), add_special_tokens=False)[0] for d in range(10)]
    comma_id = _get_pre_tokenizer().encode(",", add_special_tokens=False)[0]
    space_id = _get_pre_tokenizer().encode(" ", add_special_tokens=False)[0]
    close_id = _get_pre_tokenizer().encode("]", add_special_tokens=False)[0]

    # Build tree using digit sequence candidates.
    # Each digit is a single token, so "1234, 567]" = 1,2,3,4,,,space,5,6,7,]
    # Generate representative sequences covering all digit transitions + separators.
    all_draft = []
    for d in range(10):
        # Digit followed by all possible next digits (gives depth-2 branches)
        for d2 in range(10):
            all_draft.append(f"{d}{d2}")
        # Digit followed by separator or close
        all_draft.append(f"{d}, ")
        all_draft.append(f"{d}]")
    # Separator followed by digit
    for d in range(10):
        all_draft.append(f", {d}")

    tokenzied_draft = _get_pre_tokenizer()(all_draft, add_special_tokens=False)['input_ids']
    tokenzied_draft = [[begin,] + ids for ids in tokenzied_draft]

    one_matches_tree, _ = find_all_matches_kmp(
        tokenzied_draft,
        [begin,],
        compute_lps([begin,]),
        10,
    )
    return one_matches_tree

CACHED_BBOX_MODE_ONE_MATCHES_TREE = None

def get_bbox_mode_candidate(begin):
    '''
    Candidate nodes for bbox mode
    '''
    global CACHED_BBOX_MODE_ONE_MATCHES_TREE
    if CACHED_BBOX_MODE_ONE_MATCHES_TREE is None:
        CACHED_BBOX_MODE_ONE_MATCHES_TREE = get_bbox_mode_one_matches_tree()
    CACHED_BBOX_MODE_ONE_MATCHES_TREE.value = begin
    
    return CACHED_BBOX_MODE_ONE_MATCHES_TREE

def _block_text_to_markdown(text):
    """Convert block-level plain text to Markdown format matching page-level JSON output.

    Block outputs use raw formatting (•, —, no # prefix) but the model's JSON
    text fields use Markdown (*, -, # prefix). This conversion improves
    speculative matching for text_mode in the last block.
    """
    lines = text.split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # em-dash bullets → Markdown list
        if stripped.startswith('— ') or stripped.startswith('– '):
            stripped = '- ' + stripped[2:]
        elif stripped.startswith('—') or stripped.startswith('–'):
            stripped = '- ' + stripped[1:]
        # bullet → Markdown list
        elif stripped.startswith('• '):
            stripped = '* ' + stripped[2:]
        elif stripped.startswith('•'):
            stripped = '* ' + stripped[1:]
        out.append(stripped)
    return '\n'.join(out)

def get_category_mode_candidate(begin):
    return None


def primarily_clean_draft(draft_text):
    '''
    Clean draft text: strip extra whitespace.
    '''
    draft_text = draft_text.strip()

    # collapse repeated spaces
    draft_text = re.sub(r'\s{2,}', ' ', draft_text)

    # (padding after punctuation is not needed for LLM outputs)
    draft_text = re.sub(r'([,.:;?!])(?!\s|$)', r'\1 ', draft_text)

    draft_text = draft_text.replace(". ", ".")
    draft_text = draft_text.replace("[ ", "[")
    draft_text = draft_text.replace("] ", "]")

    # (space removal around brackets is not needed for LLM outputs)

    # (slash spacing is not needed for LLM outputs)

    # (percent spacing is not needed for LLM outputs)

    return draft_text.strip()
############################### helpers ###############################


DOTS_VLM_MAX_IMAGES = 200


class DotsOCRForCausalLM(Qwen2ForCausalLM):
    config_class = DotsOCRConfig

    def __init__(self, config: DotsOCRConfig):
        super().__init__(config)

        if isinstance(self.config.vision_config, dict):
            vision_config = DotsVisionConfig(**self.config.vision_config)
            self.config.vision_config = vision_config
        else:
            vision_config = self.config.vision_config

        self.vision_tower = DotsVisionTransformer(vision_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.FloatTensor] = None,
        img_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            assert img_mask is not None
            if grid_thw.shape[0] > DOTS_VLM_MAX_IMAGES:
                print(
                    f"Num image exceeded: {grid_thw.shape[0]} > {DOTS_VLM_MAX_IMAGES}, which may cause FSDP hang"
                )

            vision_embeddings = self.vision_tower(pixel_values, grid_thw)

            true_indices = torch.nonzero(img_mask).squeeze()
            if len(true_indices) > vision_embeddings.size(0):
                print(
                    f"img_mask sum > VE and will be truncated, mask.sum()={len(true_indices)} {vision_embeddings.size(0)=}"
                )
                true_indices = true_indices[: vision_embeddings.size(0)]
                new_img_mask = torch.zeros_like(img_mask, device=img_mask.device)
                new_img_mask[true_indices[:, 0], true_indices[:, 1]] = True
            else:
                new_img_mask = img_mask

            assert (
                vision_embeddings.size(0) == new_img_mask.sum()
            ), f"{vision_embeddings.size(0)=}, {new_img_mask.sum()=}"

            inputs_embeds = inputs_embeds.masked_scatter(
                new_img_mask.to(inputs_embeds.device).unsqueeze(-1).expand_as(inputs_embeds),
                vision_embeddings.to(inputs_embeds.device).type(inputs_embeds.dtype),
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert len(input_ids) >= 1, f"empty input_ids {input_ids.shape=} will cause gradnorm nan"
        if inputs_embeds is None:
            img_mask = input_ids == self.config.image_token_id
            inputs_embeds = self.prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw, img_mask)

        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache if use_cache is not None else self.config.use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            logits_to_keep=logits_to_keep,
            **loss_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        if cache_position is None or cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @torch.no_grad()
    def batch_generate(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pointer: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        match_window_size=3,
        match_window_size_upper_bound=10,
        max_tree_depth=15,
        max_tree_len=256,
        min_input_length=128,
        max_input_length=128,
        tokenizer: Optional[object] = None,
        draft: Optional[List] = None,
        enable_tree_decode: bool = False,
        generation_config = None,
        **kwargs,
    ) -> torch.LongTensor:
        _hsd_sync()
        st_surround = time.time()
        original_attn_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = 'flex_attention'
        if _HSD_MAX_TREE_LEN > 0:
            max_tree_len = _HSD_MAX_TREE_LEN
        # stopping conditions
        eos_token_id = kwargs.get('eos_token_id', None)
        max_new_tokens = kwargs.get('max_new_tokens', None)
        assert eos_token_id is not None, 'eos_token_id must be provided for speculative generation.'
        assert max_new_tokens is not None, 'max_new_tokens must be provided for speculative generation.'

        # Use raw (pre-cleanup) block_content for tokenization so tokens match model output.
        # primarily_clean_draft converts full-width→half-width (：→:) which mismatches model's tokens.
        _draft_items = list(draft)  # keep original dicts (with block_bbox etc.)
        draft_raw = [item.get("raw_block_content", item["block_content"]) for item in draft]
        draft = [item["block_content"] for item in draft]
        norm_draft = [to_halfwidth_clean(item) for item in draft]
        tokenized_draft = [tokenizer.encode(raw_item, add_special_tokens=False) for raw_item in draft_raw]

        decode_times = 0
        parser = StreamingTextWatcher()
        pad_token_id = tokenizer.pad_token_id

        tree_thresh = float(os.environ.get("HSD_TREE_THRESH", "1.0"))
        spec_thresh = float(os.environ.get("HSD_SPEC_THRESH", "1.0"))
        # Allow env var override for match_window_size experimentation
        _mw_override = int(os.environ.get("HSD_MATCH_WINDOW", "0"))
        if _mw_override > 0:
            match_window_size = _mw_override

        new_tokens_count = 0
        num_items = len(pointer)
        decode_modes = ["normal"] * num_items
        match_window_sizes = [match_window_size] * num_items
        matches_length = [0] * num_items
        one_matches_trees = [None] * num_items
        generated_tokens_increment = [0] * num_items
        generated_tokens = [[] for _ in range(num_items)]
        all_matches = [[] for _ in range(num_items)]
        # valid token count per block from previous iter; avoids O(block_len)
        # Python sum() loop. None on first iter → fall back to sum().
        block_valid_counts = [None] * num_items

        # Acceptance tracking
        _accept_stats = {
            "total_proposed": 0,
            "total_accepted": 0,
            "mode_counts": {"normal": 0, "speculative": 0, "tree": 0},
            "spec_proposed": 0,
            "spec_accepted": 0,
            "tree_proposed": 0,  # max possible from tree
            "tree_accepted": 0,
            # Last block parser mode tracking
            "last_select_count": 0, "last_select_tree_tokens": 0,
            "last_bbox_count": 0, "last_bbox_tree_tokens": 0,
            "last_text_count": 0, "last_text_spec_acc": 0, "last_text_spec_prop": 0,
            "last_text_tree_tokens": 0,
        }

        # Phase timing accumulators (always accumulated; cheap since _hsd_sync only syncs when HSD_PROFILE=1)
        phase_verify_loop = 0.0
        phase_cat_inputs = 0.0
        phase_mask_build = 0.0
        phase_kv_compact = 0.0
        phase_reindex_mask = 0.0

        # Pre-allocate reusable masks (avoid per-iteration allocation)
        _normal_causal_mask = torch.zeros(
            (1, 1, min_input_length, min_input_length),
            dtype=torch.bool, device=attention_mask.device,
        )
        _normal_causal_mask[0, 0, 0, 0] = True

        _SPEC_TRIL_SIZE = 2048
        _spec_tril_mask = torch.tril(torch.ones(
            (1, 1, _SPEC_TRIL_SIZE, _SPEC_TRIL_SIZE),
            dtype=torch.bool, device=attention_mask.device,
        ))

        # Cache for finished blocks' text-mode data (avoid repeated decode/encode/clean)
        _text_mode_block_cache = {}

        # first forward consumes the packed inputs directly
        model_inputs = BatchFeature({
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        })

        _hsd_sync()
        et_surround = time.time()
        if _HSD_VERBOSE:
            print(f"surround time: {et_surround - st_surround:.4f}s", flush=True)

        total_surround_time = et_surround - st_surround
        total_prepare_cpu_time = 0.0
        total_forward_time = 0.0
        total_outside_forward_time = 0.0
        total_kv_compact_time = 0.0
        first_forward_time = 0.0
        kv_compact_calls = 0
        kv_compact_skipped = 0

        while True:
            decode_times += 1

            # forward
            _hsd_sync()
            st1 = time.time()
            outputs = self.forward(**model_inputs)
            logits = outputs.logits
            pass  # forbidden_char_id removed per user request

            # Fused argmax: single GPU kernel instead of per-block argmax
            all_next_tokens = logits.argmax(dim=-1)  # (1, Q)
            _hsd_sync()
            et1 = time.time()
            if _HSD_VERBOSE:
                print("attention mask shape", attention_mask.shape, flush=True)
                print(f"forward time: {et1 - st1:.4f}s", flush=True)
            iter_forward_time = et1 - st1
            total_forward_time += iter_forward_time
            if decode_times == 1:
                first_forward_time = iter_forward_time

            _hsd_sync()
            st_outside_forward = time.time()

            # CPU batch snapshot: one GPU->CPU transfer, then entire per-block loop runs on Python
            _hsd_sync()
            _t_snap = time.time()
            pointer_list = pointer.tolist()
            all_next_tokens_list = all_next_tokens[0].tolist()
            input_ids_list = input_ids[0].tolist()
            position_ids_list = position_ids[0].tolist()
            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] cpu_snapshot: {time.time()-_t_snap:.4f}s", flush=True)

            new_input_ids = []
            new_position_ids = []
            new_pointer_list = [0] * num_items  # Python list avoids GPU scalar writes
            new_attention_mask = [None] * num_items
            attention_mask_filterd_mask = torch.zeros(
                [attention_mask.shape[-1]], dtype=torch.bool, device=attention_mask.device
            )
            input_ids_in_attn_mask_start = attention_mask.shape[-1] - attention_mask.shape[-2]

            # per-block decode bookkeeping
            block_start = 0
            next_token_int = 0
            num_blocks = num_items
            _t_block_loop = time.time()

            for block_id in range(num_blocks):
                block_end = pointer_list[block_id] + 1
                mode = decode_modes[block_id]

                if mode == "normal":
                    # valid_length: use cached count from previous iter (avoids Python loop)
                    _bvc = block_valid_counts[block_id]
                    valid_length = _bvc if _bvc is not None else \
                        sum(1 for t in input_ids_list[block_start:block_end] if t != pad_token_id)
                    new_ptr = block_start + valid_length - 1
                    pointer_list[block_id] = new_ptr

                    next_token_int = all_next_tokens_list[new_ptr]
                    generated_tokens[block_id].append(next_token_int)
                    generated_tokens_increment[block_id] = 1
                    new_tokens_count += 1
                    block_start = block_end

                    if matches_length[block_id] in [0, 1]:
                        match_window_sizes[block_id] = match_window_size
                    else:
                        match_window_sizes[block_id] += 1

                elif mode == "speculative":
                    _bvc = block_valid_counts[block_id]
                    valid_length = _bvc if _bvc is not None else \
                        sum(1 for t in input_ids_list[block_start:block_end] if t != pad_token_id)
                    new_ptr = block_start + valid_length - 1
                    pointer_list[block_id] = new_ptr

                    next_token_group_list = all_next_tokens_list[block_start:block_end]
                    reference_list = input_ids_list[block_start:block_end]
                    block_len = block_end - block_start

                    # Speculative acceptance via logit ratio.
                    # logit_ref / logit_pred >= thresh (when pred logit > 0)
                    # Falls back to exact match when pred logit <= 0.
                    _effective_thresh = spec_thresh
                    if _effective_thresh >= 1.0:
                        # Fast path: exact match only (no GPU ops needed)
                        acessment = [
                            next_token_group_list[i] == reference_list[i + 1]
                            for i in range(block_len - 1)
                        ]
                    else:
                        # Step 1: CPU exact-match (zero overhead, same as st=1.0)
                        acessment = [
                            next_token_group_list[i] == reference_list[i + 1]
                            for i in range(block_len - 1)
                        ]
                        # Step 2: Find first CPU mismatch
                        _first_miss = len(acessment)
                        for _fi, _ok in enumerate(acessment):
                            if not _ok:
                                _first_miss = _fi
                                break
                        # Step 3: Logit ratio check from first mismatch onward
                        # logit_ref / logit_pred >= thresh (only when pred logit > 0)
                        if _first_miss < len(acessment):
                            _start_idx = block_start + _first_miss
                            _end_idx = block_start + len(acessment)
                            _range = torch.arange(_start_idx, _end_idx, device=logits.device)
                            _ref_toks = input_ids[0, _start_idx+1: _end_idx+1]
                            _pred_toks = all_next_tokens[0, _start_idx: _end_idx]
                            _ref_l = logits[0, _range, _ref_toks]
                            _pred_l = logits[0, _range, _pred_toks]
                            # Only apply logit ratio when pred logit > 0; otherwise exact match
                            _gpu_pass = ((_ref_l / _pred_l) >= _effective_thresh) & (_pred_l > 0)
                            _gpu_pass = _gpu_pass.tolist()
                            for _gi, _gv in enumerate(_gpu_pass):
                                acessment[_first_miss + _gi] = acessment[_first_miss + _gi] or _gv

                    # First mismatch
                    diff_loc = len(acessment)
                    for i, ok in enumerate(acessment):
                        if not ok:
                            diff_loc = i
                            break

                    # Accepted tokens: restore reference for accepted positions
                    accepted_tokens = list(next_token_group_list[:diff_loc + 1])
                    for i in range(diff_loc):
                        accepted_tokens[i] = reference_list[i + 1]
                    generated_tokens[block_id].extend(accepted_tokens)
                    generated_tokens_increment[block_id] = diff_loc + 1
                    new_tokens_count += diff_loc + 1
                    _accept_stats["spec_proposed"] += block_len - 1  # number of tokens checked
                    _accept_stats["spec_accepted"] += diff_loc  # number accepted (excluding the new model token)
                    # Debug: log first few mismatches to understand rejection patterns
                    if _HSD_VERBOSE and diff_loc < block_len - 1 and _accept_stats.get("_mismatch_logged", 0) < 10:
                        _accept_stats["_mismatch_logged"] = _accept_stats.get("_mismatch_logged", 0) + 1
                        model_tok = tokenizer.decode([next_token_group_list[diff_loc]])
                        draft_tok = tokenizer.decode([reference_list[diff_loc + 1]])
                        ctx = tokenizer.decode(reference_list[max(0, diff_loc-2):diff_loc+1])
                        print(f"  [mismatch] blk={block_id} pos={diff_loc}/{block_len-1} "
                              f"model='{model_tok}' draft='{draft_tok}' ctx='{ctx}'", flush=True)
                    next_token_int = next_token_group_list[diff_loc]

                    if matches_length[block_id] in [0, 1]:
                        match_window_sizes[block_id] = match_window_size
                    else:
                        match_window_sizes[block_id] += diff_loc + 1

                    if diff_loc != block_len - 1:
                        pointer_list[block_id] = block_start + diff_loc
                        attention_mask_filterd_mask[
                            input_ids_in_attn_mask_start + block_start + diff_loc + 1:
                            input_ids_in_attn_mask_start + block_end
                        ] = True

                    block_start = block_end

                elif mode == "tree":
                    _bvc = block_valid_counts[block_id]
                    valid_length = _bvc if _bvc is not None else \
                        sum(1 for t in input_ids_list[block_start:block_end] if t != pad_token_id)
                    pointer_list[block_id] = block_start + valid_length - 1

                    input_ids_in_attn_mask_start = \
                        attention_mask.shape[-1] - attention_mask.shape[-2]

                    next_token_group_list = all_next_tokens_list[block_start:block_end]
                    now_processed_node = one_matches_trees[block_id]

                    decode_token = next_token_group_list[0]
                    ### patch: GPU-side logits-ratio fallback (rare; disabled at tree_thresh >= 1.0)
                    if tree_thresh < 1.0 and parser.text_mode and (decode_token not in now_processed_node.children):
                        children_tokenizer_id = list(now_processed_node.children.keys())
                        children_logits = logits[0, block_start: block_end, :][0, children_tokenizer_id]
                        prob_ref = logits[0, block_start: block_end, :][0, decode_token]
                        logits_ratio = children_logits / prob_ref
                        if (logits_ratio >= tree_thresh).any():
                            best_idx = torch.argmax(logits_ratio).item()
                            decode_token = children_tokenizer_id[best_idx]
                    ### patch
                    generated_tokens[block_id].append(decode_token)
                    generated_tokens_increment[block_id] = 1
                    new_tokens_count += 1

                    attention_mask_filterd_mask[
                        input_ids_in_attn_mask_start + block_start
                    ] = True

                    new_index = 0
                    tree_decode_num = 0

                    while decode_token in now_processed_node.children and \
                        now_processed_node.children[decode_token].input_ids_index != -1:
                        now_processed_node = now_processed_node.children[decode_token]

                        new_index = now_processed_node.input_ids_index
                        now_processed_node.input_ids_index = -1
                        decode_token = next_token_group_list[new_index]
                        ### patch (disabled at tree_thresh >= 1.0)
                        if tree_thresh < 1.0 and parser.text_mode and (decode_token not in now_processed_node.children):
                            children_tokenizer_id = list(now_processed_node.children.keys())
                            children_logits = logits[0, block_start: block_end, :][new_index, children_tokenizer_id]
                            prob_ref = logits[0, block_start: block_end, :][new_index, decode_token]
                            logits_ratio = children_logits / prob_ref
                            if (logits_ratio >= tree_thresh).any():
                                best_idx = torch.argmax(logits_ratio).item()
                                decode_token = children_tokenizer_id[best_idx]
                        ### patch
                        generated_tokens[block_id].append(decode_token)
                        generated_tokens_increment[block_id] += 1
                        tree_decode_num += 1

                        attention_mask_filterd_mask[
                            input_ids_in_attn_mask_start + block_start + new_index
                        ] = True
                    one_matches_trees[block_id].input_ids_index = -1

                    pointer_list[block_id] = block_start + new_index
                    attention_mask_filterd_mask[
                        input_ids_in_attn_mask_start + block_start:
                        input_ids_in_attn_mask_start + block_end
                    ] = ~attention_mask_filterd_mask[
                        input_ids_in_attn_mask_start + block_start:
                        input_ids_in_attn_mask_start + block_end
                    ]

                    new_tokens_count += tree_decode_num
                    _accept_stats["tree_accepted"] += tree_decode_num
                    next_token_int = next_token_group_list[new_index]

                    if matches_length[block_id] in [0, 1]:
                        match_window_sizes[block_id] = match_window_size
                    else:
                        match_window_sizes[block_id] += 1 + tree_decode_num

                    block_start = block_end

                elif mode == "finished":
                    block_start = block_end
                    continue

                # stopping conditions
                if generated_tokens[block_id][-1] in eos_token_id:
                    attention_mask_filterd_mask = \
                        attention_mask_filterd_mask | (attention_mask[0, 0, pointer_list[block_id], :] > -0.1)
                    decode_modes[block_id] = "finished"
                    generated_tokens[block_id] = generated_tokens[block_id][:-1]
                    continue
                if (block_id != num_blocks - 1) and \
                    len(
                        find_all_matches_kmp(
                            [generated_tokens[block_id]],
                            generated_tokens[block_id][-10:],
                            compute_lps(generated_tokens[block_id][-10:]),
                            1
                        )[1]
                    ) > 10 and (20993 not in generated_tokens[block_id]):
                    attention_mask_filterd_mask = \
                        attention_mask_filterd_mask | (attention_mask[0, 0, pointer_list[block_id], :] > -0.1)
                    generated_tokens[block_id] = tokenized_draft[block_id]
                    decode_modes[block_id] = "finished"
                    continue

                # find draft matches
                _bbox_used_spec = False
                if block_id == num_blocks - 1:
                    new_chunk = tokenizer.decode(generated_tokens[block_id][-generated_tokens_increment[block_id]:])
                    parser.feed(new_chunk.encode('utf-8'))
                this_match_window_size = min(
                    match_window_size_upper_bound, match_window_sizes[block_id]
                )
                generated_window_list = generated_tokens[block_id][-this_match_window_size:]
                if block_id != num_blocks - 1:
                    one_matches_tree, matches = find_all_matches_kmp(
                        [tokenized_draft[block_id]],
                        generated_window_list,
                        compute_lps(generated_window_list),
                        max_tree_depth
                    )
                else:
                    # category mode merged into select mode
                    _last_parser_mode = "select"
                    if not (parser.bbox_mode or parser.text_mode):
                        matches = []
                        one_matches_tree = get_select_mode_candidate(begin=generated_tokens[block_id][-1])
                        _accept_stats["last_select_count"] += 1
                    if parser.bbox_mode:
                        _last_parser_mode = "bbox"
                        matches = []
                        one_matches_tree = get_bbox_mode_candidate(begin=generated_tokens[block_id][-1])
                        _accept_stats["last_bbox_count"] += 1

                    if parser.text_mode:
                        _last_parser_mode = "text"
                        _accept_stats["last_text_count"] += 1
                        generated_window_text = tokenizer.decode(generated_window_list)
                        generated_window_text = unescape_string_final(generated_window_text)
                        generated_window_text_clean = clean_string_and_get_maps(generated_window_text)['cleaned_string']
                        generated_window_text_clean = [c for c in generated_window_text_clean]

                        # Text mode with caching for finished blocks
                        if len(generated_tokens) > 1:
                            temp_draft = []
                            temp_tokenized_draft = []
                            temp_draft_clean = []
                            temp_new_to_old_map = []
                            uncached_bids = []
                            uncached_tokens = []
                            for bid in range(len(generated_tokens)-1):
                                if bid in _text_mode_block_cache:
                                    c = _text_mode_block_cache[bid]
                                    temp_draft.append(c[0])
                                    temp_tokenized_draft.append(c[1])
                                    temp_draft_clean.append(c[2])
                                    temp_new_to_old_map.append(c[3])
                                else:
                                    uncached_bids.append(bid)
                                    uncached_tokens.append(generated_tokens[bid])
                                    temp_draft.append(None)
                                    temp_tokenized_draft.append(None)
                                    temp_draft_clean.append(None)
                                    temp_new_to_old_map.append(None)
                            if uncached_bids:
                                decoded_list = tokenizer.batch_decode(uncached_tokens, skip_special_tokens=True)
                                for j, bid in enumerate(uncached_bids):
                                    one_draft = decoded_list[j]
                                    one_draft = one_draft.replace("\n\n", "\n")
                                    temp_draft[bid] = one_draft
                                    temp_tokenized_draft[bid] = tokenizer.encode(json.dumps(one_draft, ensure_ascii=False)[1:-1], add_special_tokens=False)
                                    one_draft_clean = clean_string_and_get_maps(one_draft)
                                    temp_draft_clean[bid] = [c for c in one_draft_clean['cleaned_string']]
                                    temp_new_to_old_map[bid] = one_draft_clean['new_to_old_map']
                                    if decode_modes[bid] == "finished":
                                        _text_mode_block_cache[bid] = (one_draft, temp_tokenized_draft[bid], temp_draft_clean[bid], temp_new_to_old_map[bid])
                        else:
                            one_draft = norm_draft[0] if len(norm_draft) > 0 else ""
                            one_draft = one_draft.replace("\n\n", "\n")
                            temp_draft = [norm_draft[0], ] if len(norm_draft) > 0 else [""]
                            temp_tokenized_draft = [tokenizer.encode(json.dumps(one_draft, ensure_ascii=False)[1:-1], add_special_tokens=False), ]
                            one_draft_clean = clean_string_and_get_maps(one_draft)
                            temp_draft_clean = [[c for c in one_draft_clean['cleaned_string']], ]
                            temp_new_to_old_map = [one_draft_clean['new_to_old_map'], ]

                        if len(generated_window_text_clean) != 0:
                            _, matches = find_all_matches_kmp(
                                temp_draft_clean,
                                generated_window_text_clean,
                                compute_lps(generated_window_text_clean),
                                1,
                            )
                        else:
                            matches = []

                        if len(matches) != 1:
                            one_matches_tree, _ = find_all_matches_kmp(
                                temp_tokenized_draft,
                                generated_window_list,
                                compute_lps(generated_window_list),
                                max_tree_depth
                            )
                        else:
                            one_matches_tree = empty_node
                all_matches[block_id] = matches
                one_matches_trees[block_id] = one_matches_tree
                matches_length[block_id] = len(matches)

                # prepare new inputs (CPU-side positions from snapshot)
                cur_pos_base = position_ids_list[pointer_list[block_id]] + 1
                if len(matches) == 1:
                    item_idx, start, end = matches[0]
                    if block_id != num_blocks - 1:
                        start += len(generated_window_list) - 1
                        this_input_ids = tokenized_draft[block_id][start: end]
                        if _HSD_MAX_SPEC_LEN > 0 and len(this_input_ids) > _HSD_MAX_SPEC_LEN:
                            this_input_ids = this_input_ids[:_HSD_MAX_SPEC_LEN]
                        if not _HSD_TIGHT_PACK:
                            this_input_ids += [pad_token_id] * (math.ceil(
                                len(this_input_ids) / min_input_length
                            ) * min_input_length - len(this_input_ids))
                        new_input_ids.append(
                            torch.tensor([this_input_ids],
                            dtype=torch.long,
                            device=input_ids.device),
                        )
                    else:
                        start += len(generated_window_text_clean) - 1
                        start = temp_new_to_old_map[item_idx][start]

                        one_draft = temp_draft[item_idx][start + 1:]
                        one_draft = json.dumps(one_draft, ensure_ascii=False)[1:-1]
                        might_overlap = tokenizer.decode(generated_tokens[block_id][-1])
                        if one_draft.startswith(might_overlap):
                            one_draft = one_draft[len(might_overlap):]
                        this_input_ids = [generated_tokens[block_id][-1],] + \
                            tokenizer.encode(one_draft, add_special_tokens=False)
                        if _HSD_MAX_SPEC_LEN > 0 and len(this_input_ids) > _HSD_MAX_SPEC_LEN:
                            this_input_ids = this_input_ids[:_HSD_MAX_SPEC_LEN]
                        if not _HSD_TIGHT_PACK:
                            this_input_ids += [pad_token_id] * (math.ceil(
                                len(this_input_ids) / min_input_length
                            ) * min_input_length - len(this_input_ids))
                        new_input_ids.append(
                            torch.tensor(
                                [this_input_ids],
                                dtype=torch.long,
                                device=input_ids.device
                            )
                        )
                        end = start + new_input_ids[-1].shape[-1]

                    this_len = new_input_ids[-1].shape[1]
                    new_position_ids.append(
                        torch.arange(cur_pos_base, cur_pos_base + this_len,
                                     device=position_ids.device).unsqueeze(0)
                    )
                    new_pointer_list[block_id] = this_len
                    # Use pre-allocated tril mask (view, no copy)
                    if this_len <= _SPEC_TRIL_SIZE:
                        new_attention_mask[block_id] = _spec_tril_mask[:, :, :this_len, :this_len]
                    else:
                        new_attention_mask[block_id] = torch.tril(torch.ones(
                            (1, 1, this_len, this_len),
                            dtype=torch.bool, device=attention_mask.device,
                        ))

                    block_valid_counts[block_id] = len(this_input_ids) - (
                        this_input_ids.count(pad_token_id) if not _HSD_TIGHT_PACK else 0
                    )
                    decode_modes[block_id] = "speculative"
                else:
                    ## tree decode
                    if enable_tree_decode and len(one_matches_tree.children) > 0:
                        this_input_ids = []
                        this_position_ids = []
                        ancestor_indices = []
                        path_to_index = {}
                        current_index = 0

                        _tree_cap = min(max_tree_len, _HSD_MAX_SPEC_LEN) if _HSD_MAX_SPEC_LEN > 0 else max_tree_len
                        queue = deque([one_matches_tree])
                        while queue and len(this_input_ids) < _tree_cap:
                            node = queue.popleft()
                            node.input_ids_index = current_index
                            path_to_index[node.path] = current_index

                            this_input_ids.append(node.value)
                            this_position_ids.append(node.depth + cur_pos_base)

                            current_ancestors = [current_index]
                            if len(node.path) > 2:
                                parent_path = node.path[:-1]
                                parent_index = path_to_index.get(parent_path)
                                if parent_index is not None:
                                    current_ancestors.extend(ancestor_indices[parent_index])

                            ancestor_indices.append(current_ancestors)
                            current_index += 1
                            for child in node.children.values():
                                queue.append(child)

                        if not _HSD_TIGHT_PACK:
                            this_input_ids += [pad_token_id] * ((math.ceil(
                                len(this_input_ids) / min_input_length
                            ) * min_input_length - len(this_input_ids)))
                            this_position_ids += [0] * (len(this_input_ids) - len(this_position_ids))

                        num_nodes = len(this_input_ids)
                        causal_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=input_ids.device)

                        row_coords = [i for i, ancestors in enumerate(ancestor_indices) for _ in ancestors]
                        col_coords = [ancestor for ancestors in ancestor_indices for ancestor in ancestors]

                        causal_mask[row_coords, col_coords] = True
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                        new_attention_mask[block_id] = causal_mask
                        new_input_ids.append(
                            torch.tensor([this_input_ids], dtype=torch.long, device=input_ids.device)
                        )
                        new_position_ids.append(
                            torch.tensor([this_position_ids], dtype=torch.long, device=position_ids.device)
                        )
                        new_pointer_list[block_id] = new_input_ids[-1].shape[1]

                        block_valid_counts[block_id] = current_index  # valid tree nodes (before padding)
                        decode_modes[block_id] = "tree"
                    ## normal decode (use pre-allocated mask)
                    else:
                        if _HSD_TIGHT_PACK:
                            new_input_ids.append(
                                torch.tensor([[next_token_int]], dtype=torch.long, device=input_ids.device)
                            )
                            new_position_ids.append(
                                torch.tensor([[cur_pos_base]], dtype=torch.long, device=position_ids.device)
                            )
                            new_pointer_list[block_id] = 1
                            new_attention_mask[block_id] = _normal_causal_mask[:, :, :1, :1]
                        else:
                            this_input_ids_list = [next_token_int] + [pad_token_id] * (min_input_length - 1)
                            this_position_ids_list = [cur_pos_base] + [0] * (min_input_length - 1)
                            new_input_ids.append(
                                torch.tensor([this_input_ids_list], dtype=torch.long, device=input_ids.device)
                            )
                            new_position_ids.append(
                                torch.tensor([this_position_ids_list], dtype=torch.long, device=position_ids.device)
                            )
                            new_pointer_list[block_id] = min_input_length
                            new_attention_mask[block_id] = _normal_causal_mask
                        block_valid_counts[block_id] = 1  # normal mode always has 1 valid token
                        decode_modes[block_id] = "normal"

            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] block_loop: {time.time()-_t_block_loop:.4f}s", flush=True)
            phase_verify_loop += time.time() - _t_block_loop

            for bid in range(num_blocks):
                if decode_modes[bid] not in ("finished",):
                    _accept_stats["mode_counts"][decode_modes[bid]] = _accept_stats["mode_counts"].get(decode_modes[bid], 0) + 1
            # Track last block's decode result per parser mode
            last_bid = num_blocks - 1
            if decode_modes[last_bid] not in ("finished",) and hasattr(parser, 'text_mode'):
                _pm = _last_parser_mode if 'block_id' in dir() and block_id == last_bid else None
                if _pm and decode_modes[last_bid] == "tree":
                    key = f"last_{_pm}_tree_tokens"
                    if key in _accept_stats:
                        _accept_stats[key] += generated_tokens_increment[last_bid]
                elif _pm == "text" and decode_modes[last_bid] == "speculative":
                    _accept_stats["last_text_spec_acc"] += generated_tokens_increment[last_bid] - 1
                    _accept_stats["last_text_spec_prop"] += block_valid_counts[last_bid] - 1 if block_valid_counts[last_bid] else 0

            # generation finished
            if new_tokens_count >= max_new_tokens or \
                decode_modes.count("finished") == len(decode_modes):
                break
            if decode_modes[-1] == "finished":
                break

            # Write pointer_list back to GPU tensor
            pointer = torch.tensor(pointer_list, dtype=torch.long, device=attention_mask.device)

            # Mark padding positions in the current Q as filtered from KV.
            # Note: the all(0) scan over the full attention_mask that used to appear here
            # is redundant — soft-killed columns already have all-(-inf) rows so live_cols
            # will be False for them; pad KV columns are also all-(-inf) for active blocks.
            # Keeping only the cheap pad-filter line for safety.
            attention_mask_filterd_mask[-input_ids.shape[1]:] = attention_mask_filterd_mask[-input_ids.shape[1]:] | \
                (input_ids[0] == pad_token_id)

            _hsd_sync()
            st = time.time()
            past_key_values = outputs.past_key_values
            # Build new_pointer as GPU tensor from Python list
            new_pointer = torch.tensor(new_pointer_list, dtype=pointer.dtype, device=pointer.device)
            new_pointer = new_pointer.cumsum(dim=0) - 1

            input_ids = torch.cat(new_input_ids, dim=1)
            position_ids = torch.cat(new_position_ids, dim=1)
            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] cat_input: {time.time()-st:.4f}s", flush=True)
            phase_cat_inputs += time.time() - st

            # Tight packing: pad total Q to multiple of _HSD_Q_BLOCK for flex_attention warmup compatibility
            if _HSD_TIGHT_PACK:
                actual_q_len = input_ids.shape[1]
                padded_q_len = ((actual_q_len + _HSD_Q_BLOCK - 1) // _HSD_Q_BLOCK) * _HSD_Q_BLOCK
                if padded_q_len > actual_q_len:
                    _pad_n = padded_q_len - actual_q_len
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((1, _pad_n), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
                    ], dim=1)
                    position_ids = torch.cat([
                        position_ids,
                        torch.zeros((1, _pad_n), dtype=position_ids.dtype, device=position_ids.device)
                    ], dim=1)

            # GPU-side attention_mask_indices (avoid per-block .item() syncs)
            _t_aidx = time.time()
            new_block_lengths = torch.empty_like(new_pointer)
            new_block_lengths[0] = new_pointer[0] + 1
            new_block_lengths[1:] = new_pointer[1:] - new_pointer[:-1]
            attention_mask_indices = torch.repeat_interleave(pointer, new_block_lengths)
            if attention_mask_indices.shape[0] < input_ids.shape[1]:
                attention_mask_indices = torch.cat([
                    attention_mask_indices,
                    pointer[-1:].expand(input_ids.shape[1] - attention_mask_indices.shape[0]),
                ])
            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] attn_idx: {time.time()-_t_aidx:.4f}s", flush=True)

            # Use torch.full for -inf (avoids double allocation of ones * -inf)
            _t_blkdiag = time.time()
            new_q_len = input_ids.shape[1]
            final_attention_mask_last_part = torch.full(
                (1, 1, new_q_len, new_q_len),
                float('-inf'),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

            new_attention_mask_temp = [
                item for item in new_attention_mask if item is not None
            ]
            tril_tensor = torch.block_diag(
                *[item[0, 0, :, :] for item in new_attention_mask_temp]
            )
            final_attention_mask_last_part[0, 0, :tril_tensor.shape[0], :tril_tensor.shape[1]][tril_tensor] = 0
            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] block_diag+fill: {time.time()-_t_blkdiag:.4f}s  q={new_q_len}", flush=True)
            phase_mask_build += time.time() - _t_blkdiag

            # ── Fused KV compaction + reorder ──────────────────────────────────
            # Original code did TWO passes over all KV layers:
            #   Pass-1: filter ~attention_mask_filterd_mask   (index_select per layer)
            #   Pass-2: reorder live columns to contiguous    (scatter-gather per layer)
            # We fuse them into ONE index_select by computing keep_indices directly
            #
            # Additionally, with _HSD_COMPACT_PERIOD > 1 we skip the physical
            # index_select on intermediate iterations and only do a cheap masked_fill.
            # flex_attention's BlockMask efficiently ignores fully-masked KV blocks.
            # ───────────────────────────────────────────────────────────────────────
            is_active_gpu = torch.tensor(
                [m != "finished" for m in decode_modes],
                dtype=torch.bool, device=pointer.device
            )
            active_pointer = pointer[is_active_gpu]

            relevant_masks = attention_mask.squeeze(0).squeeze(0).index_select(0, active_pointer)
            live_cols = (relevant_masks > -0.1).any(dim=0)        # [KV_len] bool
            keep_mask = live_cols & ~attention_mask_filterd_mask   # [KV_len] bool
            kv_len   = attention_mask.shape[-1]
            num_keep = int(keep_mask.sum().item())
            num_dead = kv_len - num_keep

            # Decide whether to do physical compaction this iteration.
            should_compact = (
                _HSD_COMPACT_PERIOD > 0
                and (
                    _HSD_COMPACT_PERIOD <= 1
                    or (decode_times % _HSD_COMPACT_PERIOD == 0)
                    or (num_dead > kv_len * 0.2)   # force compact when >20% dead
                )
            )

            _hsd_sync()
            st_kv = time.time()
            kv_compact_calls += 1
            if should_compact and num_keep < kv_len:
                # Physical compaction: single index_select fuses old pass-1 + pass-2.
                keep_indices = keep_mask.nonzero(as_tuple=True)[0]
                # transformers <= 4.53 exposes key_cache/value_cache lists;
                # >= 4.54 stores per-layer DynamicLayer objects with .keys/.values.
                if hasattr(past_key_values, "key_cache"):
                    for idx in range(len(past_key_values.key_cache)):
                        if past_key_values.key_cache[idx].numel():
                            past_key_values.key_cache[idx] = \
                                past_key_values.key_cache[idx].index_select(-2, keep_indices)
                            past_key_values.value_cache[idx] = \
                                past_key_values.value_cache[idx].index_select(-2, keep_indices)
                else:
                    for layer in past_key_values.layers:
                        if layer.keys is not None and len(layer.keys):
                            layer.keys = layer.keys.index_select(-2, keep_indices)
                            layer.values = layer.values.index_select(-2, keep_indices)
                attention_mask = attention_mask.index_select(-1, keep_indices)
            elif not should_compact:
                # Cheap path: mark dead columns as -inf in-place; KV cache not resized.
                dead_cols = ~keep_mask
                attention_mask = attention_mask.masked_fill(
                    dead_cols[None, None, None, :], float('-inf')
                )
                kv_compact_skipped += 1
            _hsd_sync()
            total_kv_compact_time += time.time() - st_kv

            if _HSD_VERBOSE:
                print(f"kv compact time: {time.time() - st_kv:.4f}s "
                      f"(compact={should_compact}, dead={num_dead}/{kv_len})", flush=True)

            ## attention mask
            _t_catmask = time.time()
            final_attention_mask = torch.cat(
                [attention_mask[:, :, attention_mask_indices, :],
                 final_attention_mask_last_part],
                dim=-1
            )
            if _HSD_PROFILE:
                _hsd_sync(); print(f"  [prof] cat_mask: {time.time()-_t_catmask:.4f}s  kv={final_attention_mask.shape[-1]}", flush=True)
            phase_reindex_mask += time.time() - _t_catmask

            attention_mask = final_attention_mask
            pointer = new_pointer

            model_inputs = BatchFeature({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            })
            _hsd_sync()
            et = time.time()
            if _HSD_VERBOSE:
                print(f"prepare next input time: {et - st:.4f}s", flush=True)

            _hsd_sync()
            et_outside_forward = time.time()
            if _HSD_VERBOSE:
                print(f"outside forward next input time: {et_outside_forward - st_outside_forward:.4f}s", flush=True)
            total_outside_forward_time += et_outside_forward - st_outside_forward

        self.model.config._attn_implementation = original_attn_impl

        # Print acceptance and phase timing summary
        print(f"\n=== HSD Acceptance Stats ===", flush=True)
        print(f"  Decode iterations: {decode_times}", flush=True)
        print(f"  Mode counts: {_accept_stats['mode_counts']}", flush=True)
        print(f"  Spec: {_accept_stats['spec_accepted']}/{_accept_stats['spec_proposed']} accepted"
              f" ({100*_accept_stats['spec_accepted']/max(1,_accept_stats['spec_proposed']):.1f}%)", flush=True)
        print(f"  Tree: {_accept_stats['tree_accepted']} extra tokens accepted", flush=True)
        total_gen = sum(len(g) for g in generated_tokens)
        print(f"  Total generated tokens: {total_gen}", flush=True)
        print(f"  Tokens per iteration: {total_gen/max(1,decode_times):.2f}", flush=True)
        print(f"\n=== Last Block Parser Mode Stats ===", flush=True)
        print(f"  select mode: {_accept_stats['last_select_count']}x, "
              f"tree_tokens={_accept_stats['last_select_tree_tokens']}, "
              f"avg={_accept_stats['last_select_tree_tokens']/max(1,_accept_stats['last_select_count']):.1f}/step", flush=True)
        print(f"  bbox mode:   {_accept_stats['last_bbox_count']}x, "
              f"tree_tokens={_accept_stats['last_bbox_tree_tokens']}, "
              f"avg={_accept_stats['last_bbox_tree_tokens']/max(1,_accept_stats['last_bbox_count']):.1f}/step", flush=True)
        print(f"  text mode:   {_accept_stats['last_text_count']}x, "
              f"spec_acc={_accept_stats['last_text_spec_acc']}/{_accept_stats['last_text_spec_prop']}, "
              f"tree_tokens={_accept_stats['last_text_tree_tokens']}, "
              f"avg_tree={_accept_stats['last_text_tree_tokens']/max(1,_accept_stats['last_text_count']):.1f}/step", flush=True)
        print(f"\n=== Phase Timing ===", flush=True)
        print(f"  Forward:       {total_forward_time:.3f}s ({100*total_forward_time/max(0.001,total_forward_time+total_outside_forward_time):.1f}%)", flush=True)
        print(f"  Verify loop:   {phase_verify_loop:.3f}s", flush=True)
        print(f"  Cat inputs:    {phase_cat_inputs:.3f}s", flush=True)
        print(f"  Mask build:    {phase_mask_build:.3f}s", flush=True)
        print(f"  KV compact:    {total_kv_compact_time:.3f}s", flush=True)
        print(f"  Reindex mask:  {phase_reindex_mask:.3f}s", flush=True)
        print(f"  Outside fwd:   {total_outside_forward_time:.3f}s ({100*total_outside_forward_time/max(0.001,total_forward_time+total_outside_forward_time):.1f}%)", flush=True)

        return generated_tokens, decode_times, (
            total_forward_time,
            total_outside_forward_time,
            total_surround_time,
            total_prepare_cpu_time,
            first_forward_time,        # index [4]: compat with reruncalfirstforwardtime format
            total_kv_compact_time,     # index [5]
            kv_compact_calls,          # index [6]
            kv_compact_skipped,        # index [7]
        )


    # only input_ids inputs are supported
    # (one_matches_tree mutations are expected to carry over)
    def prepare_inputs_for_tree_decode(
        self,
        one_matches_tree,
        past_key_values,
        attention_mask,
        max_tree_len=256,
        **kwargs
    ):
        # tree node depth -> packed input_ids index
        path_value_to_input_ids_index = {((-1, ), -1): 0}

        model_inputs = {}

        position_ids = []
        input_ids = []
        attention_mask = attention_mask[:, :, :, :]

        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
        else:
            cache_length = past_key_values[0][0].shape[2]


        # BFS over the tree
        queue = deque([one_matches_tree])
        while queue and len(input_ids) < max_tree_len:
            node = queue.popleft()
            # record its packed position
            node.input_ids_index = len(input_ids)
            path_value_to_input_ids_index[(node.path, node.path[-1])] = \
                len(input_ids)
            # inherit the father row of the attention mask
            attention_mask[:, :, len(input_ids), :] = \
                attention_mask[:, :, 
                path_value_to_input_ids_index[(node.path[:-1], node.path[-2])], :]
            # each node also sees itself
            attention_mask[:, :, len(input_ids), len(input_ids)] = 0
            input_ids.append(node.value)
            position_ids.append(node.depth + cache_length)

            # iterate over child nodes
            for child in node.children.values():
                queue.append(child)

        if len(input_ids) < 128:
            # pad input_ids below 128 up to 128
            input_ids += [1] * (128 - len(input_ids))
            position_ids += [position_ids[-1] + 1] * (128 - len(position_ids))

        attention_mask = attention_mask[:, :, :len(input_ids), :len(input_ids)]
        attention_mask = torch.cat(
            [
                torch.zeros(
                    (1, 1, len(input_ids), cache_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                ),
                attention_mask,
            ],
            dim=-1,
        )

        #     f"Path value to input ids index length {len(path_value_to_input_ids_index) - 1} does not match input ids length {len(input_ids)}."

        model_inputs.update(
            {
                "input_ids": torch.LongTensor([input_ids]).to(attention_mask.device),
                "position_ids": torch.LongTensor([position_ids]).to(attention_mask.device),
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        
        return model_inputs


    def prepare_inputs_for_generation_for_flash_attn(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.get_seq_length()
                max_cache_length = past_key_values.get_seq_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs
    

    # only input_ids inputs are supported
    def prepare_inputs_for_generation_for_flex_attn(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
            else:
                cache_length = past_key_values[0][0].shape[2]
        else:
            cache_length = 0

        if inputs_embeds is not None and past_key_values is None:
            _, q_len, _ = inputs_embeds.shape
            attention_mask = attention_mask[:, :, :q_len, :q_len]

            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            _, q_len = input_ids.shape
            new_k_v_len = q_len + cache_length
            attention_mask = attention_mask[:, :, cache_length: new_k_v_len, :new_k_v_len]

            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


    def warm_up_attention(self, q_range=None, k_range=(3000, 22000)):
        if q_range is None:
            if _HSD_TIGHT_PACK and _HSD_MAX_SPEC_LEN > 0:
                # With TIGHT_PACK + capped proposals, max Q is predictable:
                # num_blocks * MAX_SPEC_LEN, padded to 128. E.g. 5*32=160→256.
                # Only need a few Q values → ultra-fast warm_up.
                _max_q = ((_HSD_MAX_SPEC_LEN * 10 + 127) // 128) * 128 + 128  # generous margin
                q_range = (128, min(_max_q, 1024))
            elif _HSD_TIGHT_PACK:
                q_range = (128, 1536)
            else:
                q_range = (128, 2560)
        if os.environ.get("HSD_FAST_WARMUP", "0") == "1":
            q_range = (128, min(q_range[1], 1024))
            k_range = (3000, 12000)
        st = time.time()

        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_cache_size_limit = 1024

        with torch.no_grad():
            # build a causal mask
            # mask shape: (batch_size, num_heads, seq_len, seq_len)
            MAX_SEQ_LEN = 26000
            # arange 0..MAX_SEQ_LEN
            attention_mask = torch.arange(0, MAX_SEQ_LEN, dtype=torch.int64, device=next(self.parameters()).device)
            attention_mask = attention_mask[None, :] > attention_mask[:, None]
            attention_mask = torch.zeros(
                (MAX_SEQ_LEN, MAX_SEQ_LEN),
                dtype=next(self.parameters()).dtype,
                device=next(self.parameters()).device
            ).masked_fill(attention_mask, float('-inf'))[None, None, :, :]
            original_attn_impl = self.config._attn_implementation
            self.config._attn_implementation = 'flex_attention'

            for k_len in range(k_range[0], k_range[1], 128):
                for q_len in range(q_range[0], q_range[1], 128):
                    print(f"Warm up attention test: k_len={k_len}, q_len={q_len}", flush=True)
                    input_ids = torch.zeros(
                        (1, k_len),
                        dtype=torch.long,
                        device=next(self.parameters()).device
                    )

                    model_inputs = self.prepare_inputs_for_generation_for_flex_attn(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True
                    )

                    torch.cuda.synchronize()
                    st1 = time.time()
                    outputs = self.forward(
                        **model_inputs,
                        output_hidden_states=False,
                    )
                    torch.cuda.synchronize()
                    et1 = time.time()

                    past_key_values = outputs.past_key_values

                    input_ids = [1,] * q_len
                    input_ids = torch.tensor([input_ids],
                                             dtype=torch.long,
                                             device=next(self.parameters()).device)


                    model_inputs = self.prepare_inputs_for_generation_for_flex_attn(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask
                    )

                    model_inputs["attention_mask"] = model_inputs["attention_mask"].contiguous()

                    torch.cuda.synchronize()
                    st1 = time.time()
                    outputs = self.forward(
                        **model_inputs,
                        output_hidden_states=False,
                    )
                    torch.cuda.synchronize()
                    et1 = time.time()

            self.config._attn_implementation = original_attn_impl
        
        et = time.time()
        print(f"warm_up_attention time: {et - st:.2f} seconds", flush=True)
