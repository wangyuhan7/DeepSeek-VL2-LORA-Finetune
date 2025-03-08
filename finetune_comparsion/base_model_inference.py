# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """
    Load PIL images from the paths specified in the conversations.

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages.

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.
    """
    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            try:
                pil_img = PIL.Image.open(image_path)
                pil_img = pil_img.convert("RGB")
                pil_images.append(pil_img)
                print(f"Successfully loaded image: {image_path}")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    return pil_images


def main():
    # Create argument parser
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-vl2-tiny",
                        help="model name or local path to the model")
    parser.add_argument("--chunk_size", type=int, default=2,
                        help="chunk size for the model for prefilling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    args = parser.parse_args()
    
    dtype = torch.bfloat16

    # specify the path to the model
    model_path = args.model_path
    print(f"Loading model from: {model_path}")
    
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()
    print("Model loaded successfully!")

    # Modified conversation example with the provided image paths
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n<image>\nDescribe these two images in detail and compare them.",
            "images": [
                "/root/autodl-tmp/DeepSeek-VL2/coco_2014_caption/10037.jpg",
                "/root/autodl-tmp/DeepSeek-VL2/coco_2014_caption/100319.jpg"
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    print(f"len(pil_images) = {len(pil_images)}")
    
    if len(pil_images) == 0:
        print("No images were loaded. Please check the image paths.")
        return

    prepare_inputs = vl_chat_processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device, dtype=dtype)

    with torch.no_grad():
        print("Starting inference...")
        if args.chunk_size == -1:
            print("Using standard prefilling")
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            past_key_values = None
        else:
            print(f"Using incremental prefilling with chunk size {args.chunk_size}")
            # incremental_prefilling when using 40G GPU for vl2-small
            inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=args.chunk_size
            )

        # run the model to get the response
        print("Generating response...")
        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,

            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,

            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        print("=" * 40)
        print("MODEL OUTPUT:")
        print(f"{prepare_inputs['sft_format'][0]}", answer)
        print("=" * 40)

        # Save the generated output to a file
        with open("output_result.txt", "w") as f:
            f.write(answer)
        print("Output saved to output_result.txt")


if __name__ == "__main__":
    main()