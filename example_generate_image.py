"""Example script to generate an image and save it locally."""

import base64
from io import BytesIO

import requests
from PIL import Image


def generate_and_save_image(
    prompt: str,
    output_path: str = "generated_image.png",
    api_url: str = "http://localhost:8000/generate-cover",
):
    """
    Generate an image from a prompt and save it locally.

    Args:
        prompt: Text description of the image to generate
        output_path: Path where to save the image
        api_url: URL of the API endpoint
    """
    try:
        print(f"Generating image with prompt: '{prompt}'...")

        # Make API request
        response = requests.post(
            api_url, json={"prompt": prompt}, timeout=60  # Timeout after 60 seconds
        )

        # Check for errors
        response.raise_for_status()

        # Parse response
        data = response.json()

        # Decode base64 image
        image_data = base64.b64decode(data["image"])
        image = Image.open(BytesIO(image_data))

        # Save image
        image.save(output_path)

        print(f"✓ Image saved successfully to: {output_path}")
        print(f"  Image size: {image.size[0]}x{image.size[1]} pixels")
        print(f"  Generation time: {data['generation_time']} seconds")
        print(f"  Model used: {data['model']}")

        # Optionally open the image
        try:
            image.show()
            print("  Image opened in default viewer")
        except Exception:
            print("  (Could not open image automatically)")

        return image

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running:")
        print("  python -m app.main")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The image generation took too long.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    import sys

    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "A beautiful sunset over mountains"

    # Get output path from command line or use default
    output_path = sys.argv[2] if len(sys.argv) > 2 else "generated_image.png"

    generate_and_save_image(prompt, output_path)
