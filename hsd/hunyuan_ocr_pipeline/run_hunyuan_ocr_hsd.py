"""Run HSD (Hierarchical Speculative Decoding) with HunyuanOCR on a directory
of document images, using pre-generated pipeline drafts (PP-StructureV3 output).

Example:
    python hsd/hunyuan_ocr_pipeline/run_hunyuan_ocr_hsd.py \
        --model-path ./weights/HunyuanOCR \
        --img-dir <OmniDocBench_v1_5>/images \
        --ocr-dir ocr_results/omnidocbench_v1_5 \
        --result-dir outputs/hunyuan_ocr_hsd
"""

import os

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import argparse
import json
import sys
import time

# allow running as `python hsd/hunyuan_ocr_pipeline/run_hunyuan_ocr_hsd.py` without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoProcessor, GenerationConfig
from transformers.feature_extraction_utils import BatchFeature

from hsd.hunyuan_ocr_pipeline.modeling_hunyuan_vl_hsd import HunYuanVLForConditionalGeneration
from hsd.utils import primarily_clean_draft, clean_repeated_substrings, get_image


def get_position_ids(model, inputs):
    """Return the mrope position_ids for a processor output batch.

    transformers >= 5.13 HunYuanVL processors no longer emit `position_ids`
    (nor `token_type_ids`, renamed to `mm_token_type_ids`); recompute them via
    the model's get_rope_index as the official generate() path does.
    """
    if "position_ids" in inputs:
        return inputs["position_ids"], inputs.get("token_type_ids")
    token_type_ids = inputs.get("mm_token_type_ids", inputs.get("token_type_ids"))
    position_ids, _ = model.model.get_rope_index(
        input_ids=inputs["input_ids"],
        mm_token_type_ids=token_type_ids,
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=inputs["attention_mask"],
    )
    return position_ids, token_type_ids

DEFAULT_PROMPT = (
    "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
    "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
)


def inference(img_path, prompt, model, processor,
              generation_config=None, draft=None,
              enable_tree_decode=False):
    """HSD inference: pack the per-block draft inputs plus the full-page input
    into one batch, then let the model verify the drafts speculatively."""

    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    messages = [messages1]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs = get_image(img_path)

    def prepare_for_parallel_input():
        final_input_ids = []
        final_token_type_ids = []
        final_position_ids = []
        final_pointer = []
        final_pixel_values = []
        final_image_grid_thw = []

        if len(draft) > 1:
            for item in draft:
                block_bbox = item["block_bbox"]
                block_bbox = [
                    int(block_bbox[0] - 5),
                    int(block_bbox[1] - 5),
                    int(block_bbox[2] + 5),
                    int(block_bbox[3] + 5),
                ]

                image_slice = image_inputs.crop((block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]))
                inputs = processor(
                    text=texts,
                    images=image_slice,
                    padding=True,
                    return_tensors="pt",
                )
                position_ids, token_type_ids = get_position_ids(model, inputs)

                final_input_ids.append(inputs["input_ids"])
                final_position_ids.append(position_ids)  # b x 4 x L
                final_pointer.append(inputs["input_ids"].shape[1])
                final_token_type_ids.append(token_type_ids)
                final_image_grid_thw.append(inputs["image_grid_thw"])
                final_pixel_values.append(inputs["pixel_values"])

        # the full-page input goes last: page-level verification (Stage 2)
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        position_ids, token_type_ids = get_position_ids(model, inputs)
        final_input_ids.append(inputs["input_ids"])
        final_position_ids.append(position_ids)  # b x 4 x L
        final_pointer.append(inputs["input_ids"].shape[1])
        final_token_type_ids.append(token_type_ids)
        final_image_grid_thw.append(inputs["image_grid_thw"])
        final_pixel_values.append(inputs["pixel_values"])

        final_input_ids = torch.cat(final_input_ids, dim=-1)
        final_position_ids = torch.cat(final_position_ids, dim=-1)
        final_image_grid_thw = torch.cat(final_image_grid_thw, dim=0)
        final_pixel_values = torch.cat(final_pixel_values, dim=0)
        final_pointer = torch.cumsum(torch.tensor(final_pointer, dtype=torch.long), dim=0) - 1

        # each block attends only to its own tokens (block-diagonal causal mask)
        causal_mask = torch.ones((final_pointer[-1] + 1, final_pointer[-1] + 1), dtype=torch.bool)
        attention_mask_start = 0
        for pointer in final_pointer:
            attention_mask_end = pointer + 1
            causal_mask[attention_mask_start:attention_mask_end, attention_mask_start:attention_mask_end] = (
                ~torch.tril(torch.ones((attention_mask_end - attention_mask_start, attention_mask_end - attention_mask_start), dtype=causal_mask.dtype))
            )
            attention_mask_start = attention_mask_end
        final_attention_mask = torch.zeros((1, 1, final_input_ids.shape[1], final_input_ids.shape[1]), dtype=torch.float32)
        final_attention_mask = final_attention_mask.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        inputs = BatchFeature(
            {
                "input_ids": final_input_ids,
                "attention_mask": final_attention_mask,
                "position_ids": final_position_ids,
                "pixel_values": final_pixel_values,
                "image_grid_thw": final_image_grid_thw,
                "pointer": final_pointer,
            }
        )

        return inputs

    inputs = prepare_for_parallel_input()
    with torch.no_grad():
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        start_time = time.time()
        output_dict = model.batch_generate(
            **inputs,
            **generation_config,
            tokenizer=processor.tokenizer,
            draft=draft,
            enable_tree_decode=True,
        )
        end_times = time.time()
        output_dict["used_time"]["total_used_time"] = end_times - start_time

    generated_ids_trimmed = [output_dict["generated_tokens"][-1],]
    output_texts = clean_repeated_substrings(processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ))
    output_dict["result"] = output_texts[0]

    output_dict["decode_generated_tokens"] = processor.tokenizer.batch_decode(
        output_dict["generated_tokens"], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_dict["generated_tokens"] = json.dumps(output_dict["generated_tokens"])

    return output_dict


def load_draft(pipeline_output):
    """Load and clean the PP-StructureV3 draft blocks for one image."""
    ocr = pipeline_output["parsing_res_list"]
    for item in ocr:
        item["block_content"] = primarily_clean_draft(item["block_content"])
    ocr = [item for item in ocr if item["block_content"] != "" and item["block_content"] != " "]
    return ocr


def parse_args():
    parser = argparse.ArgumentParser(description="Run HSD acceleration with HunyuanOCR")
    # paths: env vars serve as defaults so multi-GPU wrapper scripts keep working
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "./weights/HunyuanOCR"))
    parser.add_argument("--img-dir", default=os.environ.get("IMG_DIR"),
                        help="directory of document images (env: IMG_DIR)")
    parser.add_argument("--ocr-dir", default=os.environ.get("OCR_DIR"),
                        help="directory of PP-StructureV3 draft JSONs (env: OCR_DIR)")
    parser.add_argument("--result-dir", default=os.environ.get("RESULT_DIR", "outputs/hunyuan_ocr_hsd"))
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("MAX_IMAGES", "0")),
                        help="process at most this many images (0 = all)")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the image order (default: sorted)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="generation budget; falls back to this when the model's "
                             "generation_config.json has no max_new_tokens")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.img_dir or not args.ocr_dir:
        raise SystemExit("--img-dir and --ocr-dir are required (or set IMG_DIR / OCR_DIR env vars)")
    os.makedirs(args.result_dir, exist_ok=True)

    model = HunYuanVLForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flex_attention",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.warm_up_attention()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    generation_config = GenerationConfig.from_pretrained(args.model_path).to_dict()
    generation_config = {k: generation_config[k] for k in generation_config if k in ["max_new_tokens", "eos_token_id"]}
    if not generation_config.get("max_new_tokens"):
        generation_config["max_new_tokens"] = args.max_new_tokens

    print(f"prompt: {args.prompt}", flush=True)

    img_list = sorted(os.listdir(args.img_dir))
    if args.shuffle:
        import random
        random.shuffle(img_list)
    if args.max_images > 0:
        img_list = img_list[: args.max_images]

    for img_name in img_list:
        image_path = os.path.join(args.img_dir, img_name)

        pipeline_output = json.load(
            open(os.path.join(args.ocr_dir, img_name.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json")), "r")
        )
        ocr = load_draft(pipeline_output)

        save_path = os.path.join(args.result_dir, img_name.replace(".jpg", ".json").replace(".png", ".json"))
        if os.path.exists(save_path):
            print(f"Skipping {save_path} as it already exists.", flush=True)
            continue

        print(f"Image: {image_path}", flush=True)

        res_dict = {"image": image_path}

        try:
            output_dict = inference(
                image_path, args.prompt, model, processor,
                generation_config=generation_config,
                draft=ocr, enable_tree_decode=True,
            )
            res_dict["result"] = output_dict["result"]
            res_dict["decode_times"] = output_dict["decode_times"]
            res_dict["decode_token_num"] = output_dict["decode_token_num"]
            res_dict["total_used_time"] = output_dict["used_time"]["total_used_time"]
            res_dict["raw_output"] = output_dict

            with open(save_path, "w") as f:
                json.dump(res_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing {image_path}: {e}", flush=True)
            continue


if __name__ == "__main__":
    main()
