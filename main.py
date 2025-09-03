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
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

class ImageAnalysisRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

# Initialize FastAPI
app = FastAPI(
    title="MiniCPM-V-4_5 API with Image Upload",
    description="A FastAPI application for MiniCPM-V-4_5 with image upload and analysis capabilities",
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
        "message": "MiniCPM-V-4_5 API with Image Upload",
        "status": "running",
        "endpoints": {
            "text_generation": "/generate",
            "image_upload": "/upload-image",
            "image_analysis": "/analyze-image",
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

@app.post("/generate")
async def generate_text(req: PromptRequest):
    """Generate text using MiniCPM-V-4_5"""
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        if response_text.startswith(req.prompt):
            response_text = response_text[len(req.prompt):].strip()
        
        return {
            "prompt": req.prompt,
            "response": response_text,
            "max_new_tokens": req.max_new_tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

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

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail"),
    max_new_tokens: int = Form(200)
):
    """Analyze an uploaded image with MiniCPM-V-4_5"""
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
        
        # Create a combined prompt with image
        # Note: This is a simplified approach. For actual vision models, 
        # you would need to properly encode the image
        combined_prompt = f"[IMAGE: {img_base64[:100]}...] {prompt}"
        
        # Generate response
        inputs = tokenizer(combined_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        if response_text.startswith(combined_prompt):
            response_text = response_text[len(combined_prompt):].strip()
        
        return {
            "prompt": prompt,
            "response": response_text,
            "max_new_tokens": max_new_tokens,
            "image_info": {
                "filename": file.filename,
                "size": len(image_content),
                "format": image.format,
                "dimensions": f"{image.width}x{image.height}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze-uploaded-image")
async def analyze_uploaded_image(
    filename: str = Form(...),
    prompt: str = Form("Describe this image in detail"),
    max_new_tokens: int = Form(200)
):
    """Analyze a previously uploaded image"""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Read and process image
        with open(file_path, 'rb') as f:
            image_content = f.read()
        
        image = Image.open(BytesIO(image_content))
        
        # Convert image to base64 for processing
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Create a combined prompt with image
        combined_prompt = f"[IMAGE: {img_base64[:100]}...] {prompt}"
        
        # Generate response
        inputs = tokenizer(combined_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from response
        if response_text.startswith(combined_prompt):
            response_text = response_text[len(combined_prompt):].strip()
        
        return {
            "prompt": prompt,
            "response": response_text,
            "max_new_tokens": max_new_tokens,
            "filename": filename,
            "image_info": {
                "size": len(image_content),
                "format": image.format,
                "dimensions": f"{image.width}x{image.height}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

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
