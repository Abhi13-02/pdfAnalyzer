import io
import os
import base64
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
import requests

POLLINATIONS_API_URL = "https://text.pollinations.ai/openai"

def get_description_from_pollinations(image: Image.Image) -> str:
    """
    Generate a description for an image using the Pollinations AI API.
    The image is converted to a base64-encoded JPEG.
    """
    try:
        # Convert the image to JPEG and encode in base64
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        # Build the payload for Pollinations API
        payload = {
            "model": "openai",  # Ensure this model supports vision per Pollinations documentation
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image like your describing it to a blind person. If its a written document then read it aloud in a way that the blind guy understand and follows everything and dosnt miss any detail or any line that is written:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(POLLINATIONS_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Extract the generated description text from the API response
        description = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return description.strip() if description else "No description available."
    except Exception as e:
        return f"Error generating description: {e}"

def extract_text_from_image(image) -> str:
    """Extract text from an image using OCR (Tesseract)."""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def extract_page_data(pdf_path: str):
    """
    For each page in the PDF, extract:
      - The text content from the page.
      - OCR text from the rendered page image.
      - Any images embedded in the page along with their descriptions.
    Returns a list of dictionaries, one per page.
    """
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    # Extract text from each page using PyMuPDF
    pages_text = [page.get_text("text").strip() for page in doc]
    
    # Convert entire PDF to images (one image per page)
    ocr_pages = convert_from_path(pdf_path)
    
    # Extract images per page (using PyMuPDF per page)
    pages_images = []
    for page_number in range(num_pages):
        page = doc[page_number]
        image_list = page.get_images(full=True)
        images_data = []
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                images_data.append({
                    "image_index": image_index + 1,
                    "error": f"Unable to open image: {e}"
                })
                continue

            # Get a description using the Pollinations API
            description = get_description_from_pollinations(image)
            images_data.append({
                "image_index": image_index + 1,
                "image_extension": image_ext,
                "description": description
            })
        pages_images.append(images_data)
    
    # Combine per-page data
    pages = []
    for i in range(num_pages):
        page_data = {
            "page_number": i + 1,
            "page_text": pages_text[i],
            "ocr_text": extract_text_from_image(ocr_pages[i]) if i < len(ocr_pages) else "",
            "images": pages_images[i]
        }
        pages.append(page_data)
    
    return pages
