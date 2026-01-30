import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)


HF_TOKEN = input("Введите HF-токен")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)


def describe_image(image) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
    [output_text] = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text


model = Qwen3VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype="auto",
    # device_map="auto",
    device_map={"": "cuda:0"},
    token=HF_TOKEN,
    # quantization_config=quant_config,
    trust_remote_code=True,
)

# default processer
processor = AutoProcessor.from_pretrained(
    # "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-2B-Instruct",
)

result = {}

for image_path in Path("raccoons").iterdir():
    image = Image.open(image_path).convert("RGB")
    description = describe_image(image)
    result[image_path.name] = description

Path("result.json").write_text(json.dumps(result))
