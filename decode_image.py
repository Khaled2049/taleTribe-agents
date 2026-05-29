"""Utility script to decode base64 image from API response and save it."""

import base64
import json
import sys
from io import BytesIO

from PIL import Image


def decode_base64_image(base64_string: str, output_path: str = "generated_image.png"):
    """
    Decode a base64-encoded image and save it to a file.

    Args:
        base64_string: Base64-encoded image string
        output_path: Path where to save the decoded image
    """
    try:
        # Decode base64 string
        image_data = base64.b64decode(base64_string)

        # Open image with PIL
        image = Image.open(BytesIO(image_data))

        # Save image
        image.save(output_path)
        print(f"Image saved successfully to: {output_path}")
        print(f"Image size: {image.size[0]}x{image.size[1]} pixels")
        print(f"Image format: {image.format}")

        return image

    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return None


def decode_from_json(json_file: str, output_path: str = None):
    """
    Decode image from API response JSON file.

    Args:
        json_file: Path to JSON file containing API response
        output_path: Optional output path (defaults to 'generated_image.png')
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        if "image" not in data:
            print("Error: 'image' field not found in JSON response")
            return None

        if output_path is None:
            output_path = "generated_image.png"

        return decode_base64_image(data["image"], output_path)

    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python decode_image.py <base64_string> [output_path]")
        print("  python decode_image.py --json <json_file> [output_path]")
        print("\nExamples:")
        print("  python decode_image.py 'iVBORw0KGgoAAAANSUhEUgAA...' output.png")
        print("  python decode_image.py --json response.json")
        sys.exit(1)

    if sys.argv[1] == "--json":
        if len(sys.argv) < 3:
            print("Error: Please provide JSON file path")
            sys.exit(1)
        json_file = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        decode_from_json(json_file, output_path)
    else:
        base64_string = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "generated_image.png"
        decode_base64_image(base64_string, output_path)
