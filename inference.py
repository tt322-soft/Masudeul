from unsloth import FastVisionModel
from transformers import CLIPImageProcessor
from PIL import Image
import torch
import os

def load_model():
    try:
        print("Checking CUDA availability:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA Device:", torch.cuda.get_device_name(0))
        
        # Check if lora_model exists
        lora_path = "lora_model"
        if not os.path.exists(lora_path):
            print(f"LoRA path not found: {os.path.abspath(lora_path)}")
            return None, None, None
        print(f"Found LoRA path: {os.path.abspath(lora_path)}")
        print(f"LoRA contents: {os.listdir(lora_path)}")
        
        # Load base model and tokenizer
        model_name = "unsloth/Pixtral-12B-2409"
        print(f"Loading base model: {model_name}")
        base_model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            attn_implementation="eager"
        )
        print("Base model loaded successfully")
        
        # Load LoRA adapter
        # print("Loading LoRA adapter...")
        # base_model.load_adapter(lora_path)
        # print("LoRA adapter loaded successfully")
        
        # Load image processor
        print("Loading image processor...")
        image_processor = CLIPImageProcessor.from_pretrained(
            model_name,
            do_resize=True,
            size={"height": 800, "width": 800},
            do_center_crop=True,
        )
        print("Image processor loaded successfully")
        
        return base_model, tokenizer, image_processor
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return None, None, None

def generate_response(model, tokenizer, image_processor, image_path, instruction):

    if image_path != "":
        # Load and process image
        image = Image.open(image_path)
        
        # Prepare messages
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
    
    else:
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": instruction}
            ]}
        ]
    
    # Prepare inputs
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    if image_path != "":
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
    
    else:
        inputs = tokenizer(
            input_text,
            # add_special_tokens=False,
            skip_special_tokens = True,
            return_tensors="pt",
        ).to("cuda")
    
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    
    # # Generate response
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=100,
    #     use_cache=True,
    #     temperature=1.5,
    #     min_p=0.1
    # )
    outputs = model.generate(
        **inputs,
        max_new_tokens=8192,      # Increased from 64 to allow longer responses
        # use_cache=True,
        temperature=0.5,         # Reduced from 1.5 for more focused responses
        min_p=0.1,
        do_sample=True,         # Enable sampling for more natural responses
        # top_k=50,               # Limit vocabulary to top 50 tokens
        # top_p=0.9,             # Nucleus sampling threshold
        repetition_penalty=1.2, # Prevent repetitive text
        # pad_token_id=tokenizer.pad_token_id,
        # eos_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    # Load model
    model, tokenizer, image_processor = load_model()
    
    # Example inference
    image_path = ""
    instruction = "What information do you have about BlackRock? Give me answer as query format in graph rag."
    # image_path = "test.jpg"
    # instruction = "Explain about this."
    
    response = generate_response(model, tokenizer, image_processor, image_path, instruction)
    print(f"Response: {response}")

# def generate_chunked_response(model, tokenizer, image_processor, image_path, instruction, num_chunks=3):
#     """Generate response in multiple chunks"""
#     all_responses = []
#     current_instruction = instruction

#     for _ in range(num_chunks):
#         # Generate chunk with original parameters
#         if image_path != "":
#             messages = [
#                 {"role": "user", "content": [
#                     {"type": "image"},
#                     {"type": "text", "text": current_instruction}
#                 ]}
#             ]
#         else:
#             messages = [
#                 {"role": "user", "content": [
#                     {"type": "text", "text": current_instruction}
#                 ]}
#             ]
        
#         input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
#         inputs = tokenizer(
#             input_text,
#             skip_special_tokens=True,
#             return_tensors="pt",
#         ).to("cuda")
        
#         if 'token_type_ids' in inputs:
#             inputs.pop('token_type_ids')
        
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=64,
#             temperature=0.5,
#             min_p=0.1,
#             do_sample=True,
#             repetition_penalty=1.2
#         )
        
#         chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         if not chunk.strip():
#             break
            
#         all_responses.append(chunk)
#         # Update instruction to continue the response
#         current_instruction = f"{' '.join(all_responses)} Continue:"
    
#     return " ".join(all_responses)

# def generate_response(model, tokenizer, image_processor, image_path, instruction):
#     response = generate_chunked_response(model, tokenizer, image_processor, "", instruction)

#     return response


# # Update the main execution part:
# if __name__ == "__main__":
#     model, tokenizer, image_processor = load_model()
    
#     instruction = "tell me about history of new york"
#     response = generate_chunked_response(model, tokenizer, image_processor, "", instruction)
#     print(f"Response: {response}")