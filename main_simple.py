from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
        "model_loaded": False,
        "device": "not_loaded"
    }

@app.post("/chat")
async def chat_with_model(req: ChatRequest):
    """Chat with MiniCPM-V-4_5 for text conversation"""
    return {
        "message": req.message,
        "response": f"Hello! I'm MiniCPM-V-4_5. You said: '{req.message}'. This is a demo response - the actual model is not loaded yet.",
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "note": "Model not loaded - this is a demo response"
    }

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
        
        return {
            "prompt": prompt,
            "extracted_text": f"Demo OCR result: This image contains text that would be extracted by MiniCPM-V-4_5. Original prompt: {prompt}",
            "max_new_tokens": max_new_tokens,
            "image_info": {
                "filename": file.filename,
                "size": len(image_content),
                "format": image.format,
                "dimensions": f"{image.width}x{image.height}"
            },
            "ocr_confidence": "demo",
            "note": "Model not loaded - this is a demo response"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
