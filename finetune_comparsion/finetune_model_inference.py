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

import os
import sys
import time
import argparse
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import PIL.Image
from peft import PeftModel, PeftConfig

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor


def load_pil_images(image_paths: List[str]) -> List[PIL.Image.Image]:
    """
    Load PIL images from the given paths.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        pil_images (List[PIL.Image.Image]): List of loaded PIL images.
    """
    pil_images = []

    for image_path in image_paths:
        try:
            # 获取绝对路径
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
                
            print(f"Loading image from: {image_path}")
            if not os.path.exists(image_path):
                print(f"WARNING: Image path does not exist: {image_path}")
                continue
                
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
            print(f"Successfully loaded image: {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

    return pil_images


def run_inference(
    model_path: str,
    image_paths: List[str],
    base_model_path: str = "deepseek-ai/deepseek-vl2-tiny",  # 添加基础模型路径参数
    prompt: str = "Describe these images in detail and compare them.",
    max_new_tokens: int = 512,
    temperature: float = 0.4,
    top_p: float = 0.9,
    chunk_size: int = -1,
    output_file: Optional[str] = None
):
    """
    Run inference with DeepSeek-VL2 model.
    
    Args:
        model_path: Path to the adapter/PEFT model directory.
        image_paths: List of paths to images.
        base_model_path: Path or HF ID of the base model.
        prompt: Text prompt for the model.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        chunk_size: Chunk size for incremental prefilling (-1 to disable).
        output_file: Path to save the output (None to print to console).
    """
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU may be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # 设置数据类型
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # 验证和处理模型路径
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    print(f"Loading adapter model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        return
    
    # 加载处理器（从基础模型）
    try:
        print(f"Loading processor from {base_model_path}...")
        processor = DeepseekVLV2Processor.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        print("Processor loaded successfully")
    except Exception as e:
        print(f"Error loading processor: {e}")
        return
    
    tokenizer = processor.tokenizer
    
    # 加载基础模型，然后应用PEFT适配器
    try:
        print(f"Loading base model from {base_model_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        print("Base model loaded successfully")
        
        print(f"Loading adapter from {model_path}...")
        # 检查是否存在PEFT配置文件
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        if not os.path.exists(peft_config_path):
            print(f"WARNING: No adapter_config.json found at {peft_config_path}")
            print("Searching for alternative adapter config locations...")
            
            # 尝试其他可能的配置文件名
            for config_name in ["adapter_config.json", "config.json", "peft_config.json"]:
                test_path = os.path.join(model_path, config_name)
                if os.path.exists(test_path):
                    peft_config_path = test_path
                    print(f"Found adapter config at: {peft_config_path}")
                    break
        
        # 加载PEFT模型
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            torch_dtype=dtype
        )
        print("Adapter model loaded and merged successfully")
    except Exception as e:
        print(f"Error loading adapter model: {e}")
        print("Attempting to continue with base model only...")
        try:
            model = base_model
        except:
            print("Failed to fall back to base model. Exiting.")
            return
    
    # 将模型移动到设备并设置为评估模式
    model = model.to(device).eval()
    
    # 加载图像
    pil_images = load_pil_images(image_paths)
    if not pil_images:
        print("No images were loaded. Please check the image paths.")
        return
    print(f"Loaded {len(pil_images)} images")
    
    # 构建图像标记
    image_tokens = "<image>" * len(pil_images)
    full_prompt = f"{image_tokens}\n{prompt}"
    
    # 构建对话
    conversation = [
        {
            "role": "<|User|>",
            "content": full_prompt,
            "images": image_paths,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    # 准备输入
    print("Preparing inputs...")
    start_time = time.time()
    prepare_inputs = processor.__call__(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device, dtype=dtype)
    print(f"Inputs prepared in {time.time() - start_time:.2f} seconds")
    
    # 开始推理
    with torch.no_grad():
        print("Starting inference...")
        start_time = time.time()
        
        try:
            # 根据chunk_size决定是否使用增量预填充
            if chunk_size == -1:
                print("Using standard prefilling")
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None
            else:
                print(f"Using incremental prefilling with chunk size {chunk_size}")
                # 增量预填充 (适用于内存有限的情况)
                inputs_embeds, past_key_values = model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=chunk_size
                )
            
            # 生成回应
            print(f"Generating response with max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}...")
            outputs = model.generate(
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
                max_new_tokens=max_new_tokens,
                
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                
                use_cache=True,
            )
            
            inference_time = time.time() - start_time
            print(f"Inference completed in {inference_time:.2f} seconds")
            
            # 解码输出
            answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
            
            # 输出结果
            print("\n" + "=" * 40)
            print("MODEL OUTPUT:")
            print(answer)
            print("=" * 40 + "\n")
            
            # 保存输出
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(answer)
                print(f"Output saved to {output_file}")
            
            return answer
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description="Run inference with DeepSeek-VL2 model")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the PEFT/adapter model directory")
    parser.add_argument("--base_model_path", type=str, default="deepseek-ai/deepseek-vl2-tiny",
                        help="Path or HF ID of the base model")
    parser.add_argument("--images", type=str, nargs="+", required=True,
                        help="Paths to image files")
    parser.add_argument("--prompt", type=str, default="Describe these images in detail and compare them.",
                        help="Text prompt for the model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--chunk_size", type=int, default=2,
                        help="Chunk size for incremental prefilling (-1 to disable)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the output (None to print to console)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行推理
    run_inference(
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        image_paths=args.images,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_size=args.chunk_size,
        output_file=args.output_file
    )


if __name__ == "__main__":
    main()