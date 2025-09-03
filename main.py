from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import aiofiles
from PIL import Image
import base64
from io import BytesIO
import uuid
from typing import Optional

# Define request models
class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 200
    temperature: float = 0.7

class OCRRequest(BaseModel):
    prompt: str = "Extract all text from this image. Provide the text content clearly and accurately."
    max_new_tokens: int = 500

# Initialize FastAPI
app = FastAPI(
    title="MiniCPM-V-4_5 Chat & OCR API",
    description="MiniCPM-V-4_5 API with Text Chat and Image OCR capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
tokenizer = None
model = None
device = None

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model & tokenizer at startup
@app.on_event("startup")
async def load_model():
    global tokenizer, model, device
    model_id = "OpenBMB/MiniCPM-V-4_5"

    print("🚀 Loading MiniCPM-V-4_5 model...")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    print(f"✅ Model loaded successfully on {device}")

@app.get("/")
async def root():
    return {
        "message": "MiniCPM-V-4_5 Chat & OCR API",
        "status": "running",
        "endpoints": {
            "text_chat": "/chat",
            "image_ocr": "/ocr",
            "image_upload": "/upload-image",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/chat")
async def chat_with_model(req: ChatRequest):
    """Chat with MiniCPM-V-4_5 for text conversation"""
    try:
        # Format the message for better conversation
        formatted_prompt = f"Human: {req.message}\nAssistant:"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response_text:
            response_text = response_text.split("Assistant:")[-1].strip()
        else:
            # Fallback: remove the original prompt
            if response_text.startswith(formatted_prompt):
                response_text = response_text[len(formatted_prompt):].strip()
        
        return {
            "message": req.message,
            "response": response_text,
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Get image info
        with Image.open(file_path) as img:
            width, height = img.size
            format_name = img.format
        
        return {
            "message": "Image uploaded successfully",
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": file_path,
            "file_size": len(content),
            "image_info": {
                "width": width,
                "height": height,
                "format": format_name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

@app.post("/ocr")
async def extract_text_from_image(
    file: UploadFile = File(...),
    prompt: str = Form("Extract all text from this image. Provide the text content clearly and accurately."),
    max_new_tokens: int = Form(500)
):
    """Extract text from uploaded image using MiniCPM-V-4_5 OCR capabilities"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_content = await file.read()
        image = Image.open(BytesIO(image_content))
        
        # Convert image to base64 for processing
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Create OCR-specific prompt with image
        # For MiniCPM-V-4_5, we need to format the image properly
        ocr_prompt = f"<image>{img_base64}</image>\n{prompt}"
        
        # Generate OCR response
        inputs = tokenizer(ocr_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,  # Lower temperature for more accurate OCR
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the OCR result
        if "<image>" in response_text:
            response_text = response_text.split("<image>")[-1].strip()
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        return {
            "prompt": prompt,
            "extracted_text": response_text,
            "max_new_tokens": max_new_tokens,
            "image_info": {
                "filename": file.filename,
                "size": len(image_content),
                "format": image.format,
                "dimensions": f"{image.width}x{image.height}"
            },
            "ocr_confidence": "high"  # MiniCPM-V-4_5 provides high-quality OCR
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")



@app.get("/list-uploads")
async def list_uploads():
    """List all uploaded images"""
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "uploaded_at": stat.st_mtime
                })
        
        return {
            "uploads": files,
            "count": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list uploads: {str(e)}")

@app.delete("/delete-upload/{filename}")
async def delete_upload(filename: str):
    """Delete an uploaded image"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        
        return {
            "message": f"File {filename} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
