from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
from PIL import Image
import io
import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
import multiprocessing

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

app = FastAPI(title="Pixtral Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add import for path handling
from pathlib import Path

def merge_lora_weights():
    """Merge LoRA weights with base model"""
    try:
        print("Starting LoRA weights merging...")
        model_name = "unsloth/Pixtral-12B-2409"
        lora_path = "lora_model"
        output_path = "merged_model"

        # Check if merged model already exists
        if Path(output_path).exists():
            print(f"Merged model already exists at {output_path}")
            return output_path

        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading and merging LoRA adapter...")
        model.load_adapter(lora_path)
        model = model.merge_and_unload()

        print(f"Saving merged model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return str(Path(output_path).absolute())
    except Exception as e:
        print(f"Error merging LoRA weights: {str(e)}")
        return None

def create_model():
    """Create LLM instance"""
    # First ensure merged model exists
    merged_path = merge_lora_weights()
    if not merged_path:
        print("Failed to merge LoRA weights, falling back to base model")
        merged_path = "unsloth/Pixtral-12B-2409"

    print(f"Creating LLM instance from: {merged_path}")
    return LLM(
        model=merged_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="half",
        trust_remote_code=True
    )

# Global model variable
model = None

def init_model(max_retries=3):
    """Initialize the LLM model with retry mechanism"""
    global model
    
    if model is not None:
        return model
        
    for attempt in range(max_retries):
        try:
            print(f"Loading model (attempt {attempt + 1}/{max_retries})...")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            model = create_model()
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Model loading failed, attempt {attempt + 1}: {str(e)}")
    return None

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global model
    multiprocessing.freeze_support()
    model = init_model()
    if model is None:
        raise RuntimeError("Failed to load model")

def process_chat(llm, prompt, sampling_params=None):
    """Process chat messages"""
    if sampling_params is None:
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "From now, You are a vision language model that trained after Cyberself AI Team. You were created/trained in 2025 by Cyberself Inc."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)
    return outputs[0].outputs[0].text

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Get the last message from the conversation
        last_message = request.messages[-1].content
        
        # Generate response
        response = process_chat(
            llm=model,
            prompt=last_message
        )
        
        return JSONResponse(content={
            "response": response,
            "messages": [*[m.dict() for m in request.messages], {"role": "assistant", "content": response}]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-with-image")
async def chat_with_image(
    file: UploadFile = File(...),
    message: str = None,
    conversation: List[ChatMessage] = []
):
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save temporary image
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        try:
            # Format prompt with image instruction
            image_prompt = f"Looking at this image: {message}"
            
            # Generate response
            response = process_chat(
                llm=model,
                prompt=image_prompt
            )
            
            # Update conversation
            updated_messages = [
                *[m.dict() for m in conversation],
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            
            return JSONResponse(content={
                "response": response,
                "messages": updated_messages
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":

    # Enable multiprocessing support
    multiprocessing.freeze_support()

    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
        log_level="info"
    )