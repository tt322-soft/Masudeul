# Pixtral Fine-Tuning API

A FastAPI service for inference with a fine-tuned Pixtral 12B multimodal model.

## Project Overview

This project provides an API for interacting with a fine-tuned version of the Pixtral-12B-2409 multimodal model. The API allows for:

- Text-based chat interactions
- Image-based chat interactions
- Health status checks

The project uses LoRA fine-tuning to adapt the base Pixtral model for specific use cases while maintaining efficient memory usage.

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- FastAPI
- vLLM
- Pillow
- Unsloth
- Transformers

## Directory Structure

- `app.py`: Main FastAPI application with API endpoints
- `inference.py`: Standalone inference script using Unsloth's FastVisionModel
- `test.py`: Test script for the API endpoints
- `check_env.py`: Utility script to verify the environment setup
- `lora_model/`: Directory containing the LoRA adapter weights and configuration

## Setup

1. Ensure you have a CUDA-compatible GPU with sufficient VRAM (at least 24GB recommended)
2. Install the required dependencies:
   ```
   pip install fastapi uvicorn torch vllm pillow transformers unsloth
   ```
3. Make sure the `lora_model` directory contains all required files

## Running the API

Start the server with:

```bash
python app.py
```

This will:
1. Load the base Pixtral-12B-2409 model
2. Apply the LoRA adapter weights
3. Start the FastAPI server on port 8000

## API Endpoints

### `/chat` (POST)

Text-based chat with the model.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is artificial intelligence?"}
  ]
}
```

**Response:**
```json
{
  "response": "AI response text here",
  "messages": [
    {"role": "user", "content": "What is artificial intelligence?"},
    {"role": "assistant", "content": "AI response text here"}
  ]
}
```

### `/chat-with-image` (POST)

Chat with the model using an image.

**Request:**
- Form data with:
  - `file`: Image file
  - `message`: Text message
  - `conversation`: Previous conversation history (optional)

**Response:**
```json
{
  "response": "AI response text here",
  "messages": [
    {"role": "user", "content": "Describe this image"},
    {"role": "assistant", "content": "AI response text here"}
  ]
}
```

### `/health` (GET)

Check if the model is loaded and ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Testing

Run the test script to verify the API is working:

```bash
python test.py
```

## Standalone Inference

You can also use the model directly without the API:

```bash
python inference.py
```

This loads the model and runs a sample inference with a predefined prompt.

## System Requirements

- CUDA-compatible GPU (NVIDIA) with at least 24GB VRAM
- 16GB+ system RAM
- 50GB+ free disk space for model files

## Model Information

This project uses the Pixtral-12B-2409 model fine-tuned with LoRA. The model supports:

- Text generation
- Image understanding
- Multimodal reasoning

## Troubleshooting

If you encounter issues:

1. Run `python check_env.py` to verify your environment setup
2. Ensure your GPU has enough memory
3. Check that all model files are present in the `lora_model` directory

## License

This project uses the Pixtral-12B-2409 model which has its own licensing terms. Please refer to the original model's license for usage restrictions. 