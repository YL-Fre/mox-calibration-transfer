from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"

TARGET_DIRS = [
    "day2A",
    "day2plus_Part1",
    "day3A",
]

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

all_images = []
for path in FIG_DIR.rglob("*"):
    if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        all_images.append(path)

hash_map = {}
for img in all_images:
    h = file_hash(img)
    hash_map.setdefault(h, []).append(img)

print("=" * 80)
print("Potential duplicate figures in target folders")
print("=" * 80)

for folder in TARGET_DIRS:
    target_path = FIG_DIR / folder
    print(f"\n--- {folder} ---")

    if not target_path.exists():
        print("Missing folder")
        continue

    for img in target_path.rglob("*"):
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
            continue

        h = file_hash(img)
        duplicates = [p for p in hash_map[h] if p != img]

        print(f"\n{img.relative_to(PROJECT_ROOT)}")
        if duplicates:
            print("  DUPLICATE OF:")
            for d in duplicates:
                print(f"   - {d.relative_to(PROJECT_ROOT)}")
        else:
            print("  UNIQUE")