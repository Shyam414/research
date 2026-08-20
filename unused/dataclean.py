import cv2
import os
import glob

# ========== SETTINGS ==========
IMAGE_FOLDER = "dataset"
SAVE_DIR = "total"
SKIP_LOG = "skipped_images.txt"
PROGRESS_FILE = "processed_images.txt"
LAST_FILE = "last_image.txt"

ROI_W, ROI_H = 128, 128
MOVE_STEP = 10

DISPLAY_W, DISPLAY_H = 1000, 700

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== GLOBAL ==========
x, y = 0, 0
scale = 1
img = None
w, h = 0, 0

last_x, last_y = None, None
roi_count = 0
saved_paths = []

# ========== LOAD PROGRESS ==========
processed_images = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        processed_images = set(f.read().splitlines())

last_image = None
if os.path.exists(LAST_FILE):
    with open(LAST_FILE, "r") as f:
        last_image = f.read().strip()

print(f"Processed: {len(processed_images)}")
print(f"Last image: {last_image}")

# ========== MOUSE ==========
def mouse_callback(event, mx, my, flags, param):
    global x, y, scale, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        x = int(mx / scale) - ROI_W // 2
        y = int(my / scale) - ROI_H // 2

        x = max(0, min(w - ROI_W, x))
        y = max(0, min(h - ROI_H, y))


# ========== GET ALL IMAGES IN ORDER ==========
all_images = []
subfolders = sorted([f.path for f in os.scandir(IMAGE_FOLDER) if f.is_dir()])

for folder in subfolders:
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    all_images.extend(imgs)

print(f"Total images: {len(all_images)}")

# ========== FIND START INDEX ==========
start_index = 0
if last_image and last_image in all_images:
    start_index = all_images.index(last_image)

print(f"Starting from index: {start_index}")

# ========== MAIN LOOP ==========
for i in range(start_index, len(all_images)):

    path = all_images[i]

    # skip if already processed
    if path in processed_images:
        continue

    print(f"\nImage [{i}/{len(all_images)}]: {path}")

    img = cv2.imread(path)
    if img is None:
        continue

    h, w = img.shape[:2]

    scale = min(DISPLAY_W / w, DISPLAY_H / h)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)))

    roi_count = 0
    saved_paths = []

    if last_x is not None:
        x, y = last_x, last_y
    else:
        x = w // 2 - ROI_W // 2
        y = h // 2 - ROI_H // 2

    print("""
Controls:
Mouse → Set ROI
W/A/S/D → Move
C → Capture
U → Undo
M → Next image (SAVE PROGRESS)
L → Skip
""")

    cv2.namedWindow("Move ROI")
    cv2.setMouseCallback("Move ROI", mouse_callback)

    while True:

        display = disp.copy()

        dx = int(x * scale)
        dy = int(y * scale)
        dw_box = int(ROI_W * scale)
        dh_box = int(ROI_H * scale)

        cv2.rectangle(display, (dx, dy), (dx+dw_box, dy+dh_box), (0,255,0), 2)

        cv2.imshow("Move ROI", display)

        key = cv2.waitKey(1) & 0xFF

        # MOVE
        if key == ord('w'):
            y = max(0, y - MOVE_STEP)
        elif key == ord('s'):
            y = min(h - ROI_H, y + MOVE_STEP)
        elif key == ord('a'):
            x = max(0, x - MOVE_STEP)
        elif key == ord('d'):
            x = min(w - ROI_W, x + MOVE_STEP)

        # CAPTURE
        elif key == ord('c'):
            roi = img[y:y+ROI_H, x:x+ROI_W]

            rel_path = os.path.relpath(path, IMAGE_FOLDER)
            save_folder = os.path.join(SAVE_DIR, os.path.dirname(rel_path))
            os.makedirs(save_folder, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(path))[0]

            save_path = os.path.join(
                save_folder,
                f"{base_name}_roi{roi_count}.jpg"
            )

            cv2.imwrite(save_path, roi)
            saved_paths.append(save_path)

            print(f"Saved: {save_path}")

            roi_count += 1
            last_x, last_y = x, y

        # UNDO
        elif key == ord('u'):
            if saved_paths:
                last_file = saved_paths.pop()
                if os.path.exists(last_file):
                    os.remove(last_file)
                    roi_count -= 1
                    print(f"Undo: {last_file}")
            else:
                print("Nothing to undo")

        # NEXT IMAGE (SAVE PROGRESS)
        elif key == ord('m'):

            # save processed list
            with open(PROGRESS_FILE, "a") as f:
                f.write(path + "\n")

            # save last image
            with open(LAST_FILE, "w") as f:
                f.write(path)

            print("Progress saved")

            break

        # SKIP
        elif key == ord('l'):
            with open(SKIP_LOG, "a") as f:
                f.write(path + "\n")

            print("Skipped")
            break

    cv2.destroyAllWindows()

print("\nDone!")