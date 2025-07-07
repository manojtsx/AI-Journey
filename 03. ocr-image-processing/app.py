from ocr.loader import load_images_from_folder
from ocr.reader import extract_text_from_image

def main():
    image_folder = 'images'
    image_paths = load_images_from_folder(image_folder)

    for path in image_paths:
        print(f"\nExtracting from: {path}")
        try:
            text = extract_text_from_image("images/test.jpg")
            print("Text:\n", text.strip())
        except Exception as e:
            print("Failed to process image:", e)

if __name__ == "__main__":
    main()