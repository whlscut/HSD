"""Run HSD (Hierarchical Speculative Decoding) with dots.ocr on a directory of
document images, using pre-generated pipeline drafts (PP-StructureV3 output).

Example:
    python hsd/dots_ocr_pipeline/run_dots_ocr_hsd.py \
        --model-path ./weights/dots.ocr \
        --img-dir <OmniDocBench_v1_5>/images \
        --ocr-dir ocr_results/omnidocbench_v1_5 \
        --result-dir outputs/dots_ocr_hsd
"""

import os

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import argparse
import json
import sys
import time

# allow running as `python hsd/dots_ocr_pipeline/run_dots_ocr_hsd.py` without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoProcessor, GenerationConfig
from transformers.feature_extraction_utils import BatchFeature
from qwen_vl_utils import process_vision_info

from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from hsd.dots_ocr_pipeline.modeling_dots_ocr_hsd import DotsOCRForCausalLM
from hsd.utils import primarily_clean_draft


def pre_process_bboxes(
    origin_image,
    bboxes,
    input_width,
    input_height,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11289600,
):
    assert isinstance(bboxes, list) and len(bboxes) > 0 and isinstance(bboxes[0], list)
    min_pixels = min_pixels or MIN_PIXELS
    max_pixels = max_pixels or MAX_PIXELS
    original_width, original_height = origin_image.size

    input_height, input_width = smart_resize(
        input_height, input_width, min_pixels=min_pixels, max_pixels=max_pixels
    )

    scale_x = original_width / input_width
    scale_y = original_height / input_height

    bboxes_out = []
    for bbox in bboxes:
        bbox_resized = [
            int(float(bbox[0]) / scale_x),
            int(float(bbox[1]) / scale_y),
            int(float(bbox[2]) / scale_x),
            int(float(bbox[3]) / scale_y),
        ]
        bboxes_out.append(bbox_resized)

    return bboxes_out


def get_prompt(prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
    prompt = dict_promptmode_to_prompt[prompt_mode]
    if prompt_mode == "prompt_grounding_ocr":
        assert bbox is not None
        bboxes = [bbox]
        bbox = pre_process_bboxes(
            origin_image, bboxes, input_width=image.width, input_height=image.height,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )[0]
        prompt = prompt + str(bbox)
    return prompt


# plain autoregressive inference (no HSD), kept as a reference baseline
def normal_inference(image_path, prompt_mode, model, processor,
                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                     dpi=200, generation_config=None, draft=None,
                     enable_tree_decode=False):
    origin_image = fetch_image(image_path)
    image = get_image_by_fitz_doc(origin_image, target_dpi=dpi)
    image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
    prompt = get_prompt(prompt_mode, None, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(
        **inputs,
        **generation_config,
        tokenizer=processor.tokenizer,
        draft=draft,
        enable_tree_decode=enable_tree_decode,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def inference(image_path, prompt_mode, model, processor,
              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
              dpi=200, generation_config=None, draft=None,
              enable_tree_decode=False):
    """HSD inference: pack the per-block draft inputs plus the full-page input
    into one batch, then let the model verify the drafts speculatively."""

    st = time.time()
    origin_image = fetch_image(image_path)
    origin_input_height, origin_input_width = origin_image.height, origin_image.width
    image = get_image_by_fitz_doc(origin_image, target_dpi=dpi)
    image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
    height_ratio = image.height / origin_input_height
    width_ratio = image.width / origin_input_width
    et = time.time()
    print(f"Image Preprocess Time: {et - st:.2f} seconds", flush=True)

    prompt = get_prompt(prompt_mode, None, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

    st = time.time()

    def prepare_for_parallel_input():
        final_input_ids = []
        final_position_ids = []
        final_pointer = []
        final_pixel_values = []
        final_image_grid_thw = []

        stt = time.time()
        if len(draft) > 1:
            for item in draft:
                block_bbox = item["block_bbox"]
                block_bbox = [
                    int(block_bbox[0] * width_ratio - 5),
                    int(block_bbox[1] * height_ratio - 5),
                    int(block_bbox[2] * width_ratio + 5),
                    int(block_bbox[3] * height_ratio + 5),
                ]

                image_slice = image.crop((block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]))

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_slice},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                final_input_ids.append(inputs["input_ids"])
                final_position_ids.append(torch.arange(0, inputs["input_ids"].shape[1], dtype=torch.long).unsqueeze(0))
                final_pointer.append(inputs["input_ids"].shape[1])
                final_image_grid_thw.append(inputs["image_grid_thw"])
                final_pixel_values.append(inputs["pixel_values"])

        # the full-page input goes last: page-level verification (Stage 2)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": dict_promptmode_to_prompt["prompt_layout_all_en"]},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        final_input_ids.append(inputs["input_ids"])
        final_position_ids.append(torch.arange(0, inputs["input_ids"].shape[1], dtype=torch.long).unsqueeze(0))
        final_pointer.append(inputs["input_ids"].shape[1])
        final_image_grid_thw.append(inputs["image_grid_thw"])
        final_pixel_values.append(inputs["pixel_values"])

        ett = time.time()
        print(f"Prepare for parallel input time: {ett - stt:.2f} seconds", flush=True)

        stt = time.time()
        final_input_ids = torch.cat(final_input_ids, dim=1)
        final_position_ids = torch.cat(final_position_ids, dim=1)
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
        ett = time.time()
        print(f"Finalize for parallel input time: {ett - stt:.2f} seconds", flush=True)

        torch.cuda.synchronize()
        stt = time.time()
        inputs = BatchFeature(
            {
                "input_ids": final_input_ids,
                "attention_mask": final_attention_mask,
                "position_ids": final_position_ids,
                "pixel_values": final_pixel_values,
                "image_grid_thw": final_image_grid_thw,
                "pointer": final_pointer,
            }
        ).to("cuda")
        torch.cuda.synchronize()
        ett = time.time()
        print(f"To cuda for parallel input time: {ett - stt:.2f} seconds", flush=True)

        return inputs

    inputs = prepare_for_parallel_input()
    generated_ids = model.batch_generate(
        **inputs,
        **generation_config,
        tokenizer=processor.tokenizer,
        draft=draft,
        enable_tree_decode=enable_tree_decode,
    )
    et = time.time()
    print(f"Speculative prepare_for_parallel_input + batch_generate time: {et - st:.2f} seconds", flush=True)

    return generated_ids


def load_draft(pipeline_output):
    """Load and clean the PP-StructureV3 draft blocks for one image.

    Uses `layout_det_res` / `overall_ocr_res` (when present) to recover header
    blocks that PP-StructureV3 drops from `parsing_res_list`.
    """
    ocr = pipeline_output["parsing_res_list"]
    for item in ocr:
        item["raw_block_content"] = item["block_content"]  # keep the original for tokenization
        item["block_content"] = primarily_clean_draft(item["block_content"])
    ocr = [item for item in ocr if item["block_content"] != "" and item["block_content"] != " "]

    # recover header blocks dropped from parsing_res_list
    if "layout_det_res" in pipeline_output:
        headers_loc = [i["coordinate"] for i in pipeline_output["layout_det_res"]["boxes"] if i["label"] == "header"]
        first_one = True
        for poly, rect in zip(pipeline_output["overall_ocr_res"]["dt_polys"], pipeline_output["overall_ocr_res"]["rec_texts"]):
            x0 = min([p[0] for p in poly])
            y0 = min([p[1] for p in poly])
            x1 = max([p[0] for p in poly])
            y1 = max([p[1] for p in poly])
            for header in headers_loc:
                hx0, hy0, hx1, hy1 = header
                inter_x0 = max(x0, hx0)
                inter_y0 = max(y0, hy0)
                inter_x1 = min(x1, hx1)
                inter_y1 = min(y1, hy1)
                inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
                box_area = (x1 - x0) * (y1 - y0)
                iou = inter_area / (box_area + 1e-5)
                if iou > 0.5:
                    if not first_one:
                        ocr.insert(1, {
                            "block_content": primarily_clean_draft(rect),
                            "block_bbox": [x0, y0, x1, y1],
                        })
                    else:
                        ocr.insert(0, {
                            "block_content": primarily_clean_draft(rect),
                            "block_bbox": [x0, y0, x1, y1],
                        })
                        first_one = False
    return ocr


def parse_args():
    parser = argparse.ArgumentParser(description="Run HSD acceleration with dots.ocr")
    # paths: env vars serve as defaults so multi-GPU wrapper scripts keep working
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "./weights/dots.ocr"))
    parser.add_argument("--img-dir", default=os.environ.get("IMG_DIR"),
                        help="directory of document images (env: IMG_DIR)")
    parser.add_argument("--ocr-dir", default=os.environ.get("OCR_DIR"),
                        help="directory of PP-StructureV3 draft JSONs (env: OCR_DIR)")
    parser.add_argument("--result-dir", default=os.environ.get("RESULT_DIR", "outputs/dots_ocr_hsd"))
    parser.add_argument("--max-images", type=int, default=int(os.environ.get("MAX_IMAGES", "0")),
                        help="process at most this many images (0 = all)")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the image order (default: sorted)")
    parser.add_argument("--prompt-mode", default="prompt_ocr", choices=list(dict_promptmode_to_prompt.keys()))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=24000,
                        help="generation budget; falls back to this when the model's "
                             "generation_config.json has no max_new_tokens")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.img_dir or not args.ocr_dir:
        raise SystemExit("--img-dir and --ocr-dir are required (or set IMG_DIR / OCR_DIR env vars)")
    os.makedirs(args.result_dir, exist_ok=True)

    model = DotsOCRForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
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

    prompt = dict_promptmode_to_prompt[args.prompt_mode]
    print(f"prompt: {prompt}", flush=True)

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

        save_path = os.path.join(args.result_dir, img_name.replace(".jpg", ".json").replace(".png", ".json").replace(".jpeg", ".json"))
        if os.path.exists(save_path):
            print(f"Skipping {save_path} as it already exists.", flush=True)
            continue

        print(f"Image: {image_path}", flush=True)

        res_dict = {"image": image_path}

        st = time.time()
        output_text_2, decode_times, used_times = inference(
            image_path, args.prompt_mode, model, processor,
            generation_config=generation_config, draft=ocr, enable_tree_decode=True,
        )
        et = time.time()
        output_text_2 = processor.tokenizer.batch_decode(output_text_2, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        res_dict["Speculative Infer Result"] = output_text_2
        res_dict["Speculative Infer Time"] = et - st
        res_dict["Speculative Decode Num"] = decode_times
        res_dict["Speculative Used Time"] = used_times
        res_dict["draft"] = [item["block_content"] for item in ocr]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(res_dict, f, indent=4, ensure_ascii=False)

        print("\n\n\n", flush=True)


if __name__ == "__main__":
    main()
