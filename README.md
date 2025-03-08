# DeepSeek-VL2-LORA-Finetune
### Purpose of Fine-Tuning
We fine-tune `Deepseek-VL2-Tiny` using the `ms-swift` framework with LoRA to enhance its caption generation capability on the `COCO 2014 dataset`. After fine-tuning, the model produces more detailed and accurate image descriptions compared to the base model.
## 1. Server Environment
- **GPU**: RTX 3090 24G
- **OS**: Ubuntu 22.04.3 LTS

## 2. Setting Up DeepSeek-VL2 Environment

### Step 1: Create a Virtual Environment
```bash
conda create --name DeepSeek-VL2_fine_tune python=3.10 -y
conda activate DeepSeek-VL2_fine_tune
```

### Step 2: Clone Repository
```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL2
```

### Step 3: Install Dependencies
```bash
cd DeepSeek-VL2
pip install -e .
```

## 3. Installing ms-swift
DeepSeek-VL2 has strict version dependencies with Swift. Arbitrarily modifying library versions may cause more conflicts. Follow the steps below carefully to ensure compatibility:

```bash
pip install 'ms-swift[all]==3.0.0' -U
pip install timm==1.0.9
pip install xformers==0.0.22.post7  # This will downgrade torch from 2.6.0 to 2.1.0
pip uninstall torchvision
pip install torchvision==0.16.0    # Install torchvision compatible with torch
pip install deepspeed==0.14.4
```

---

## 4. Fine-Tuning DeepSeek-VL2
This project uses the `ms-swift` framework to fine-tune the `DeepSeek-VL2-Tiny` model with the `COCO 2014` dataset.

### Framework Repository:
[ms-swift GitHub](https://github.com/modelscope/ms-swift/tree/main)

### Training Command
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model ./deepseek-ai/deepseek-vl2-tiny \
    --dataset "modelscope/coco_2014_caption:train" \
    --val_dataset "modelscope/coco_2014_caption:validation" \
    --output_dir ./deepseek/fine-tuned-model \
    --num_train_epochs 1 \
    --learning_rate 8e-5 \
    --lora_rank 8 \
    --lora_alpha 12 \
    --max_length 4096 \
    --save_only_model True \
    --eval_steps 2000 \
    --save_steps 2000 \
    --train_type lora \
    --deepspeed zero2 \
    --lazy_tokenize True \
    --per_device_train_batch_size 2 \
    --torch_dtype bfloat16 \
    --logging_steps 5
```

---

The fine-tuned model weights are stored in the `checkpoint-6000` folder after training.

## 5. Comparison Before and After Fine-Tuning

To compare the model's performance before and after fine-tuning, follow these steps:

### **Before Fine-Tuning**
```bash
python base_model_inference.py --model_path "deepseek-ai/deepseek-vl2-tiny" --chunk_size 2
```
- **Output File**: `output_result.txt`
- **Result Image**: `base_model_output.png`

### **After Fine-Tuning**
```bash
python model_test.py \
  --model_path "./deepseek/fine-tuned-model/v3-20250307-103814/checkpoint-6000" \
  --base_model_path "deepseek-ai/deepseek-vl2-tiny" \
  --images "./coco_2014_caption/10037.jpg" "./coco_2014_caption/100319.jpg" \
  --prompt "Describe these two images in detail and compare them." \
  --output_file "finetuned_result.txt"
```
- **Output File**: `finetuned_result.txt`
- **Result Image**: `finetune_model_output.png`

---

## 6. Additional Information

### **Dataset Information**
We use the `coco_2014_caption` dataset from ModelScope:
- **Training dataset**: `modelscope/coco_2014_caption:train`
- **Validation dataset**: `modelscope/coco_2014_caption:validation`

The dataset will be automatically downloaded when running the training script.

### **Model Download Instructions**
To use `DeepSeek-VL2-Tiny`, install `huggingface_hub` and run:
```bash
pip install huggingface_hub
huggingface-cli login  # Optional if the model is private
```
The model will be cached in `~/.cache/huggingface/hub` when first used.

---

## 7. Summary
This project fine-tunes `DeepSeek-VL2-Tiny` using LoRA with the `COCO 2014` dataset. By following the above steps,can successfully set up the environment, resolve dependency conflicts, and train the model while ensuring compatibility with Swift. The fine-tuned model generates more detailed and accurate image descriptions compared to the base model.

For more detailed results, check `base_model_output.png` and `finetune_model_output.png`.

