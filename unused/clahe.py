import os

import cv2


input_path = "total"
output_path = "clahe_output"
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for root, _, files in os.walk(input_path):
    for file in files:
        if not file.lower().endswith(image_extensions):
            continue

        image_file = os.path.join(root, file)
        relative_path = os.path.relpath(image_file, input_path)
        save_file = os.path.join(output_path, relative_path)
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipped: {image_file}")
            continue

        clahe_image = clahe.apply(image)
        cv2.imwrite(save_file, clahe_image)

print("CLAHE transform completed.")
