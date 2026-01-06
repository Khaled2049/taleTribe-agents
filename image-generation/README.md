# Local Image Generator API

A FastAPI-based application for generating images locally using Stable Diffusion models optimized for low-end machines. This application runs entirely on your local machine, requiring no external API calls or cloud services.

**Note:** This image generation service is now integrated into the unified server (`python/server.py`). You can run it standalone using the instructions below, or use it through the unified server via `python server.py` (see [../readme.md](../readme.md)).

## Features

- 🚀 **Fast Text-to-Image Generation**: Generate images from text prompts using Stable Diffusion
- 💻 **Low-End Machine Optimized**: Runs efficiently on CPU with minimal resource requirements
- 🔧 **Easy to Use**: Simple REST API with automatic interactive documentation
- 📦 **Self-Contained**: All processing happens locally, no external dependencies
- 🎨 **Customizable**: Configurable model parameters via environment variables

## Requirements

- Python 3.8 or higher
- At least 4GB RAM (8GB recommended)
- 5-10GB free disk space for model storage
- CPU or GPU (CUDA supported)

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd local-image-generation
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note for GPU users**: If you have an NVIDIA GPU with CUDA installed, PyTorch should automatically detect it. To verify CUDA is available, run:
   ```python
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
   
   If CUDA is not detected, you may need to install the CUDA-enabled version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   (Replace `cu118` with your CUDA version if different)

5. **Configure environment variables** (optional):
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to customize settings if needed. **The app will automatically use your GPU if CUDA is available** (default: `DEVICE=auto`).

## Usage

### Starting the Server

Run the FastAPI application:

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:

- **Interactive API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

### Generate an Image

#### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/generate-cover" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

#### Using Python

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/api/v1/generate-cover",
    json={"prompt": "A beautiful sunset over mountains"}
)

data = response.json()
image_data = base64.b64decode(data["image"])
image = Image.open(BytesIO(image_data))
image.save("generated_image.png")
image.show()  # Opens the image in your default image viewer
print(f"Image generated in {data['generation_time']} seconds")
```

**Using the helper script** (saves you from writing code):

1. Save the API response to a JSON file:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/generate-cover" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful sunset over mountains"}' > response.json
   ```

2. Decode and save the image:
   ```bash
   python decode_image.py --json response.json output.png
   ```

Or decode directly from base64 string:
```bash
python decode_image.py "iVBORw0KGgoAAAANSUhEUgAA..." output.png
```

#### Using the Interactive Docs

1. Navigate to http://localhost:8000/docs
2. Click on `/api/v1/generate-cover`
3. Click "Try it out"
4. Enter your prompt in the request body
5. Click "Execute"
6. Download the base64 image or view it directly

### Viewing Generated Images

The API returns images as base64-encoded strings. Here are several ways to view them:

#### Method 1: Using the Example Script (Easiest)

The simplest way - generates and saves the image in one command:

```bash
python example_generate_image.py "A beautiful sunset over mountains"
```

This will:
- Generate the image
- Save it as `generated_image.png`
- Automatically open it in your default image viewer
- Show generation statistics

You can also specify a custom output path:
```bash
python example_generate_image.py "A futuristic city" my_city.png
```

#### Method 2: Using the Helper Script

1. Save the API response to a file:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/generate-cover" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "A beautiful sunset"}' > response.json
   ```

2. Decode and save the image:
   ```bash
   python decode_image.py --json response.json
   ```

The image will be saved as `generated_image.png` and you can open it with any image viewer.

#### Method 3: Using Python (Quick View)

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/api/v1/generate-cover",
    json={"prompt": "A beautiful sunset"}
)

data = response.json()
image_data = base64.b64decode(data["image"])
image = Image.open(BytesIO(image_data))
image.show()  # Opens in default image viewer
image.save("my_image.png")  # Or save it
```

#### Method 4: Using Online Tools

1. Copy the base64 string from the API response
2. Visit https://base64.guru/converter/decode/image
3. Paste the base64 string and click "Decode"
4. Download the resulting image

#### Method 5: Using Command Line (Linux/Mac)

```bash
# Save base64 string to variable
BASE64_STRING="iVBORw0KGgoAAAANSUhEUgAA..."

# Decode and save
echo "$BASE64_STRING" | base64 -d > image.png
```

#### Method 6: Browser (FastAPI Docs)

1. Go to http://localhost:8000/docs
2. Use the interactive API
3. The response will show the base64 string
4. Copy it and use Method 3 above

### Health Check

Check if the API is running and the model is loaded:

```bash
curl http://localhost:8000/api/v1/health
```

## API Endpoints

### POST `/api/v1/generate-cover`

Generate an image from a text prompt.

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains"
}
```

**Response:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "prompt": "A beautiful sunset over mountains",
  "model": "stabilityai/sd-turbo",
  "generation_time": 1.23
}
```

### GET `/api/v1/health`

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model": "stabilityai/sd-turbo",
  "device": "cpu",
  "model_loaded": true
}
```

## Configuration

Configuration is managed through environment variables. Create a `.env` file based on `.env.example`:

### Model Configuration

- `MODEL_NAME`: Hugging Face model identifier (default: `stabilityai/sd-turbo`)
- `DEVICE`: Device to use (`cpu`, `cuda`, or `auto`, default: `auto`)
  - `auto`: Automatically uses CUDA if available, otherwise falls back to CPU
  - `cuda`: Force CUDA usage (falls back to CPU if not available)
  - `cpu`: Force CPU usage

### Image Generation Parameters

- `DEFAULT_WIDTH`: Default image width in pixels (default: `512`)
- `DEFAULT_HEIGHT`: Default image height in pixels (default: `512`)
- `DEFAULT_STEPS`: Number of inference steps (default: `4` for turbo models)
- `DEFAULT_GUIDANCE_SCALE`: Guidance scale (default: `0.0` for turbo models)

### Performance Settings

- `ENABLE_MODEL_CACHING`: Cache models locally (default: `true`)
- `TORCH_DTYPE`: PyTorch data type (`float32`, `float16`, or `auto`, default: `auto`)
  - `auto`: Uses `float16` for CUDA (faster, less memory) or `float32` for CPU
  - `float16`: Half precision (faster on GPU, may have slight quality loss)
  - `float32`: Full precision (better quality, slower)

### Other Settings

- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, default: `INFO`)

## Model Information

### Recommended Models for Low-End Machines

1. **stabilityai/sd-turbo** (Default)
   - Fastest option
   - Requires only 1-4 inference steps
   - Good quality for speed trade-off
   - ~1.4GB download

2. **runwayml/stable-diffusion-v1-5**
   - Better quality, slower
   - Requires 20-50 inference steps
   - More stable results
   - ~4GB download

### Model Download

Models are automatically downloaded from Hugging Face on first use. They are cached locally in the `models/` directory (if caching is enabled).

## Performance Tips

### For CPU-Only Machines

1. **Use the turbo model**: `stabilityai/sd-turbo` is optimized for speed
2. **Reduce image size**: Use 256x256 or 384x384 instead of 512x512
3. **Minimize inference steps**: 4 steps is usually sufficient for turbo models
4. **Close other applications**: Free up RAM for better performance
5. **Use smaller batch sizes**: Generate one image at a time

### For GPU Machines

1. **Automatic Detection**: The app automatically detects and uses CUDA if available (default behavior)
2. **Manual Override**: Set `DEVICE=cuda` in your `.env` file to explicitly use GPU
3. **Verify CUDA**: Check the startup logs to confirm CUDA device is being used
4. **Performance**: GPU will significantly speed up generation (5-10x faster than CPU)
5. **Memory**: Ensure you have enough VRAM (4GB+ recommended for 512x512 images)

### Expected Performance

- **CPU (Low-end)**: 5-30 seconds per image (512x512, 4 steps)
- **CPU (Mid-range)**: 2-10 seconds per image
- **GPU**: 0.5-2 seconds per image

## Troubleshooting

### Model Loading Fails

- Ensure you have sufficient disk space (5-10GB)
- Check your internet connection (models download from Hugging Face)
- Verify the model name is correct
- Check logs for detailed error messages

### Out of Memory Errors

- Reduce image dimensions (use 256x256 or 384x384)
- Close other applications
- Use a smaller model if available
- Consider using a machine with more RAM

### Slow Generation

- Use the turbo model (`stabilityai/sd-turbo`)
- Reduce inference steps to 4
- Reduce image dimensions
- Ensure you're using CPU-optimized PyTorch builds

### CUDA Errors

- If CUDA is not available, the app will automatically fall back to CPU
- To force CPU usage, set `DEVICE=cpu` in `.env`

## Project Structure

```
local-image-generation/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app initialization
│   ├── config.py               # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API route handlers
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── __init__.py
│       └── image_service.py    # Image generation logic
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Style

This project follows PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` or `pylint` for linting
- `mypy` for type checking

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion models
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework

## Support

For issues, questions, or contributions, please open an issue on the project repository.

