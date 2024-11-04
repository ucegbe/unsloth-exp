import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel
import json



# Global variables
model = None
tokenizer = None
max_seq_length = 2048  # Adjust as needed
dtype = torch.bfloat16  # Adjust as needed
load_in_4bit = True  # Adjust as needed

def model_fn(model_dir):
    global model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,  
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    return model



def input_fn(input_data, content_type):
    if content_type == 'application/json':
        input_data = json.loads(input_data)
        return input_data
    else:
        raise ValueError('Unsupported content type: {}'.format(content_type))


def predict_fn(input_data, model):
    global tokenizer
    print(input_data)
    # Assuming input_data is a dictionary with 'instruction' and 'input' keys
    instruction = input_data.get('instruction', '')
    input_text = input_data.get('input', '')

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{0}

### Input:
{1}

### Response:
{2}"""

    formatted_prompt = alpaca_prompt.format(instruction, input_text, "")

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    text_streamer = TextStreamer(tokenizer)
    output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1500)

    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the generated part (after "### Response:")
    response = response.split("### Response:")[-1].strip()
    print(response)
    return response
