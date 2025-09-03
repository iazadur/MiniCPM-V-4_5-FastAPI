# 🚀 MiniCPM-V-4_5 FastAPI with Image Upload

A complete FastAPI application that serves the MiniCPM-V-4_5 model with image upload and analysis capabilities.

## ✨ Features

- **Text Generation**: Generate text using MiniCPM-V-4_5
- **Image Upload**: Upload and store images
- **Image Analysis**: Analyze uploaded images with custom prompts
- **REST API**: Full REST API with automatic documentation
- **Web Interface**: HTML test interface for easy testing
- **CORS Support**: Ready for frontend integration

## 🛠️ Installation

1. **Clone and setup environment**:

   ```bash
   cd minicpm-fastapi
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the server**:

   ```bash
   ./start_server.sh
   ```

   Or manually:

   ```bash
   source venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🌐 API Endpoints

### Base URLs

- **API**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

### Endpoints

| Method   | Endpoint                    | Description                       |
| -------- | --------------------------- | --------------------------------- |
| `GET`    | `/`                         | API information                   |
| `GET`    | `/health`                   | Health check                      |
| `POST`   | `/generate`                 | Generate text                     |
| `POST`   | `/upload-image`             | Upload image                      |
| `POST`   | `/analyze-image`            | Analyze uploaded image            |
| `POST`   | `/analyze-uploaded-image`   | Analyze previously uploaded image |
| `GET`    | `/list-uploads`             | List all uploaded images          |
| `DELETE` | `/delete-upload/{filename}` | Delete uploaded image             |

## 📝 Usage Examples

### 1. Text Generation

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a short story about a cat", "max_new_tokens": 100}'
```

### 2. Image Upload

```bash
curl -X POST "http://127.0.0.1:8000/upload-image" \
  -F "file=@your_image.jpg"
```

### 3. Image Analysis

```bash
curl -X POST "http://127.0.0.1:8000/analyze-image" \
  -F "file=@your_image.jpg" \
  -F "prompt=Describe this image in detail" \
  -F "max_new_tokens=200"
```

## 🧪 Testing

1. **Web Interface**: Open `test_upload.html` in your browser
2. **API Documentation**: Visit http://127.0.0.1:8000/docs
3. **Health Check**: Visit http://127.0.0.1:8000/health

## 📁 Project Structure

```
minicpm-fastapi/
├── venv/                  # Virtual environment
├── uploads/               # Uploaded images (created automatically)
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── test_upload.html       # Web test interface
├── start_server.sh        # Server startup script
└── README.md             # This file
```

## 🔧 Configuration

The application automatically detects the best available device:

- **MPS** (Apple Silicon Macs)
- **CPU** (fallback)

Model settings can be adjusted in `main.py`:

- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (0.7)
- `top_p`: Nucleus sampling (0.9)

## 🚨 Important Notes

1. **First Run**: The model will be downloaded on first startup (can take several minutes)
2. **Memory**: Ensure sufficient RAM for the model (recommended 8GB+)
3. **Storage**: Model files are cached in your home directory
4. **Image Processing**: Currently uses simplified image encoding for demonstration

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure you have sufficient RAM and storage
2. **Port Already in Use**: Change port in `start_server.sh` or kill existing process
3. **Permission Denied**: Make sure `start_server.sh` is executable (`chmod +x start_server.sh`)

### Logs

Check the console output for detailed error messages and model loading progress.

## 🔮 Future Enhancements

- [ ] Proper vision model integration
- [ ] Streaming responses
- [ ] Batch processing
- [ ] Authentication
- [ ] Rate limiting
- [ ] Docker support

## 📄 License

This project is for educational and development purposes.
