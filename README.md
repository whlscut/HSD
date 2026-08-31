x# HSD: Training-Free Acceleration for Document Parsing Vision-Language Models with Hierarchical Speculative Decoding

[![arXiv](https://img.shields.io/badge/arXiv-2602.12957-b31b1b.svg)](https://arxiv.org/abs/2602.12957)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-blue)](https://eccv.ecva.net/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![dots.ocr](https://img.shields.io/badge/%F0%9F%A4%97_weights-dots.ocr-8A2BE2)](https://huggingface.co/rednote-hilab/dots.ocr)
[![HunyuanOCR](https://img.shields.io/badge/%F0%9F%A4%97_weights-HunyuanOCR-076BFD)](https://huggingface.co/tencent/HunyuanOCR)

**HSD** (Hierarchical Speculative Decoding) is a **training-free** method that accelerates end-to-end document parsing vision-language models (VLMs) by 2-7× with near-lossless quality.

- A lightweight **pipeline drafter** (PP-StructureV3) first predicts region partitions and generates coarse drafts for each region.
- **Stage 1** verifies the region-level drafts **in parallel** for efficiency.
- **Stage 2** performs page-level verification on the refined outputs to preserve full-page coherence (reading order, cross-region structure).

The implementation builds on several techniques described in the paper: **Decoupled Speculative Verification (DSV)**, prefix-tree batching, draft–target KMP matching, tight packing, periodic KV-cache compaction, and `torch.compile`-ed FlexAttention (BlockMask grid: Q×KV = 128×64).

## Repository Structure

```
hsd/
├── hsd/
│   ├── utils.py                      # shared draft cleaning helpers
│   ├── dots_ocr_pipeline/            # HSD for dots.ocr (3B)
│   │   ├── run_dots_ocr_hsd.py       # entry-point script
│   │   ├── modeling_dots_ocr_hsd.py  # DotsOCRForCausalLM with HSD batch_generate
│   │   ├── configuration_dots.py     # vendored dots.ocr config (MIT)
│   │   └── modeling_dots_vision.py   # vendored dots.ocr vision encoder (MIT)
│   └── hunyuan_ocr_pipeline/         # HSD for HunyuanOCR (0.9B)
│       ├── run_hunyuan_ocr_hsd.py    # entry-point script
│       └── modeling_hunyuan_vl_hsd.py# HunYuanVLForConditionalGeneration with HSD
├── requirements-dots.txt
└── requirements-hunyuan.txt
```

## Installation

The two pipelines depend on different `transformers` versions, so we recommend one conda environment per pipeline (Python ≥ 3.10). Both need **flash-attn** for the vision encoders and use FlexAttention for the language-model decode path.

### dots.ocr pipeline

```bash
conda create -n hsd-dots python=3.10 -y
conda activate hsd-dots
pip install -r requirements-dots.txt
# install the dots_ocr package (prompts, image utils) from upstream
# (--no-deps: upstream pins transformers==4.56.1, which conflicts with the verified combo)
pip install --no-deps git+https://github.com/rednote-hilab/dots.ocr.git
```

> `flash-attn` needs a matching CUDA build. If no prebuilt wheel fits your setup, build from source with `MAX_JOBS=$(nproc) pip install flash-attn==2.8.0.post2 --no-build-isolation`.

### HunyuanOCR pipeline

The HunyuanOCR pipeline requires a **specific transformers prerelease** (commit `82a06db03535c49aa987719ed0746a76093b1ec4`, "hunyuan vision prerelease") that ships `HunYuanVLProcessor`. Install it first:

```bash
conda create -n hsd-hunyuan python=3.10 -y
conda activate hsd-hunyuan

git clone https://github.com/huggingface/transformers.git
cd transformers
git fetch origin 82a06db03535c49aa987719ed0746a76093b1ec4
git checkout FETCH_HEAD
pip install -e .

cd .. && pip install -r requirements-hunyuan.txt
```

> transformers is Apache-2.0 licensed, so installing (and redistributing, if you prefer to vendor the checkout) from this commit is fully permitted — just keep its LICENSE file.

## Model Weights

| Model | Download | License |
|---|---|---|
| dots.ocr (3B) | [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | [dots.ocr LICENSE AGREEMENT](https://huggingface.co/rednote-hilab/dots.ocr/blob/main/LICENSE) |
| HunyuanOCR (0.9B) | [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | [Tencent Hunyuan Community License](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |

```bash
huggingface-cli download rednote-hilab/dots.ocr --local-dir weights/dots.ocr
huggingface-cli download tencent/HunyuanOCR --local-dir weights/HunyuanOCR
```

> Note the territorial restrictions in the Hunyuan weights license (EU/UK/Korea exclusion) before redistribution.

## Generating Drafts with PP-StructureV3

HSD consumes region drafts produced by a lightweight pipeline parser. We use **PP-StructureV3** from PaddleOCR:

```bash
pip install "paddleocr[doc-parser]>=3.0"   # or: pip install paddlex

# produces one <image_stem>.json per image under <save_path>
paddlex --pipeline PP-StructureV3 \
    --input <img_dir>/ \
    --device gpu:0 \
    --save_path ocr_results/
```

Each draft JSON must contain a `parsing_res_list` where every block has:

```json
{
  "parsing_res_list": [
    {"block_label": "text", "block_content": "...", "block_bbox": [x0, y0, x1, y1]},
    ...
  ]
}
```

Optional fields used by the dots.ocr pipeline for header recovery:

- `layout_det_res.boxes` — layout detection boxes with `label` / `coordinate`
- `overall_ocr_res.dt_polys` / `overall_ocr_res.rec_texts` — full-page OCR polygons and texts

The run scripts look up `<image_stem>.json` inside `--ocr-dir` for each `<image_stem>.png/jpg` in `--img-dir`.

## Quickstart

### dots.ocr

```bash
python hsd/dots_ocr_pipeline/run_dots_ocr_hsd.py \
    --model-path weights/dots.ocr \
    --img-dir <path_to_images> \
    --ocr-dir ocr_results \
    --result-dir outputs/dots_ocr_hsd
```

### HunyuanOCR

```bash
python hsd/hunyuan_ocr_pipeline/run_hunyuan_ocr_hsd.py \
    --model-path weights/HunyuanOCR \
    --img-dir <path_to_images> \
    --ocr-dir ocr_results \
    --result-dir outputs/hunyuan_ocr_hsd
```

Useful flags (both scripts):

- `--max-images N` — process only the first N images (smoke tests)
- `--shuffle` — randomize image order (default: sorted)
- `--prompt-mode` (dots.ocr only) — prompt presets from the [dots.ocr](https://github.com/rednote-hilab/dots.ocr) package (`dots_ocr.utils.prompts`)
- `--prompt` (HunyuanOCR only) — override the default Chinese parsing prompt
- `--max-new-tokens` — fallback generation budget when the model's `generation_config.json` lacks one

Each run writes one JSON per image containing the parsing result plus timing fields (`Speculative Infer Time`, `Speculative Decode Num`, ...) used to measure end-to-end speedup. Already-existing result files are skipped, so interrupted runs resume automatically.

## Evaluating on OmniDocBench v1.5

1. Download [OmniDocBench v1.5](https://github.com/opendatalab/OmniDocBench) and point `--img-dir` at its `images/` directory.
2. Generate PP-StructureV3 drafts for all images (command above).
3. Run either pipeline. The output JSONs contain the parsed markdown and per-image latency.

## Tuning Knobs

HSD behavior is controlled via environment variables (defaults work well on OmniDocBench v1.5):

### Shared

| Variable | Default | Meaning |
|---|---|---|
| `HSD_PROFILE` | `0` | `1` enables per-step CUDA-synchronized timing |
| `HSD_VERBOSE` | `0` | `1` enables verbose debug logging |
| `HSD_TIGHT_PACK` | `0` | `1` removes per-block query padding (fewer attention kernels) |
| `HSD_COMPACT_PERIOD` | `3` | physical KV-cache compaction every N iterations (1 = every step, 0 = never) |
| `HSD_MAX_SPEC_LEN` | `0` | cap on the number of draft tokens verified per step (0 = unlimited) |
| `HSD_MAX_TREE_LEN` | `0` | override the max prefix-tree size (0 = default) |
| `HSD_FAST_WARMUP` | `0` | `1` shrinks the FlexAttention warm-up grid (faster startup, may JIT during decode) |

### dots.ocr pipeline

| Variable | Default | Meaning |
|---|---|---|
| `HSD_TREE_THRESH` | `1.0` | logit-ratio acceptance threshold for tree decode (≥1 = exact match) |
| `HSD_SPEC_THRESH` | `1.0` | logit-ratio acceptance threshold for speculative verification |
| `HSD_MATCH_WINDOW` | `0` | override the draft–target matching window size |
| `HSD_TOKENIZER_PATH` | — | dots.ocr tokenizer dir, if the model dir is not next to this package |

### HunyuanOCR pipeline

| Variable | Default | Meaning |
|---|---|---|
| `MATCH_WINDOW_SIZE` | `3` | draft–target matching window size |
| `SPEC_MODE` | `fmst` | speculative matching strategy |
| `SPEC_TOPK` | `3` | number of candidate matches per position |
| `SPEC_MARGIN` | `2.0` | acceptance margin for candidate matches |
| `SPEC_THRESH` | `1` | acceptance threshold |

The first generation after model loading runs a FlexAttention warm-up (`model.warm_up_attention()`), which JIT-compiles attention kernels for a grid of sequence lengths. This one-time cost can take 20-40 minutes on the full grid; `HSD_FAST_WARMUP=1` trades coverage for startup speed and is fine for correctness testing.

## Acknowledgements

- [dots.ocr](https://github.com/rednote-hilab/dots.ocr) (rednote-hilab) — MIT License
- [HunyuanOCR](https://github.com/Tencent/HunyuanOCR) (Tencent) — Hunyuan Community License
- [PaddleOCR / PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR) — draft generation pipeline
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — evaluation benchmark
- [transformers](https://github.com/huggingface/transformers) (Apache-2.0) — the HunyuanOCR pipeline builds on the `82a06db0` prerelease commit

## Citation

```bibtex
@article{liao2026hsd,
  title={HSD: Training-Free Acceleration for Document Parsing Vision-Language Models with Hierarchical Speculative Decoding},
  author={Liao, Wenhui and Li, Hongliang and Xie, Pengyu and Cai, Xinyu and Shen, Yufan and Xin, Yi and Qin, Qi and Ye, Shenglong and Li, Tianbin and Hu, Ming and He, Junjun and Liu, Yihao and Wang, Wenhai and Dou, Min and Fu, Bin and Shi, Botian and Qiao, Yu and Jin, Lianwen},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

This repository's code is released under the **Apache-2.0** License (see [LICENSE](LICENSE)).

- dots.ocr is MIT licensed; HunyuanOCR is under the Tencent Hunyuan Community License.
- Model weights are governed by their own agreements (see [Model Weights](#model-weights)).
