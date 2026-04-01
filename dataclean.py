import cv2
import os
import glob
import random

# ========== SETTINGS ==========
IMAGE_FOLDER = "dataset"
SAVE_DIR = "roi_output"
SKIP_LOG = "skipped_images.txt"

ROI_W, ROI_H = 128, 128
MOVE_STEP = 10

DISPLAY_W, DISPLAY_H = 1000, 700

os.makedirs(SAVE_DIR, exist_ok=True)

# ========== GLOBAL (for mouse) ==========
x, y = 0, 0
scale = 1
img = None
w, h = 0, 0

# remember last ROI position (for speed)
last_x, last_y = None, None


# ========== MOUSE FUNCTION ==========
def mouse_callback(event, mx, my, flags, param):
    global x, y, scale, w, h

    if event == cv2.EVENT_LBUTTONDOWN:
        x = int(mx / scale) - ROI_W // 2
        y = int(my / scale) - ROI_H // 2

        # keep inside bounds
        x = max(0, min(w - ROI_W, x))
        y = max(0, min(h - ROI_H, y))


# ========== GET SUBFOLDERS ==========
subfolders = [f.path for f in os.scandir(IMAGE_FOLDER) if f.is_dir()]

print(f"Total folders: {len(subfolders)}")


# ========== PROCESS ==========
for folder in subfolders:

    print(f"\nProcessing folder: {folder}")

    imgs = glob.glob(os.path.join(folder, "*.jpg"))
    random.shuffle(imgs)

    for path in imgs:

        print(f"\nImage: {path}")

        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # ===== SCALE =====
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        disp = cv2.resize(img, (int(w * scale), int(h * scale)))

        # ===== INITIAL ROI =====
        if last_x is not None:
            x, y = last_x, last_y   # reuse previous position
        else:
            x = w // 2 - ROI_W // 2
            y = h // 2 - ROI_H // 2

        print("""
Controls:
Mouse Click → Set ROI position
W/A/S/D → Move ROI
C → Capture ROI
L → Skip image
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

            # ===== MOVE =====
            if key == ord('w'):
                y = max(0, y - MOVE_STEP)

            elif key == ord('s'):
                y = min(h - ROI_H, y + MOVE_STEP)

            elif key == ord('a'):
                x = max(0, x - MOVE_STEP)

            elif key == ord('d'):
                x = min(w - ROI_W, x + MOVE_STEP)

            # ===== CAPTURE =====
            elif key == ord('c'):
                roi = img[y:y+ROI_H, x:x+ROI_W]

                # maintain folder structure
                rel_path = os.path.relpath(path, IMAGE_FOLDER)
                save_folder = os.path.join(SAVE_DIR, os.path.dirname(rel_path))
                os.makedirs(save_folder, exist_ok=True)

                save_path = os.path.join(save_folder, os.path.basename(path))
                cv2.imwrite(save_path, roi)

                print(f"Saved: {save_path}")

                # remember position
                last_x, last_y = x, y

                break

            # ===== SKIP =====
            elif key == ord('l'):
                print("Skipped")

                with open(SKIP_LOG, "a") as f:
                    f.write(path + "\n")

                break

        cv2.destroyAllWindows()

    print(f"Finished folder: {folder}")

print("\nAll images processed ")