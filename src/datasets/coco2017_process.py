import glob
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the folder containing your rePOPE .jsonl files
repope_jsonl_dir = "../datasets/repope/repope_annotations"

# Path to your extracted COCO val2014 images
coco_images_dir = "../datasets/repope/val2014/"

# SAFETY SWITCH: Set to False ONLY when you are ready to delete!
DRY_RUN = False


def extract_image_id(filename):
    """
    Safely extracts the integer ID from COCO filenames.
    Handles both 'COCO_val2014_000000397133.jpg' and '000000397133.jpg' -> 397133
    """
    base_name = filename.split(".")[0]  # Remove .jpg
    number_string = base_name.split("_")[-1]  # Get the last chunk after underscores
    return int(number_string)


def prune_unused_coco_images():
    # 1. Gather all required Image IDs from ALL json files in the directory
    required_image_ids = set()
    jsonl_files = glob.glob(os.path.join(repope_jsonl_dir, "*.json"))

    if not jsonl_files:
        print(f"Error: No .json files found in {repope_jsonl_dir}")
        return

    print(f"Scanning {len(jsonl_files)} POPE json files...")
    for jsonl_file in jsonl_files:
        with open(jsonl_file) as f:
            for line in f:
                data = json.loads(line)
                img_id = extract_image_id(data["image"])
                required_image_ids.add(img_id)

    print(f"Found {len(required_image_ids)} unique images required by rePOPE.")

    # 2. Iterate through the COCO folder and check against the required IDs
    deleted_count = 0
    kept_count = 0

    all_images = [f for f in os.listdir(coco_images_dir) if f.endswith(".jpg")]
    print(f"Found {len(all_images)} total images in the COCO directory.")

    for img_filename in all_images:
        img_path = os.path.join(coco_images_dir, img_filename)
        img_id = extract_image_id(img_filename)

        if img_id not in required_image_ids:
            if not DRY_RUN:
                os.remove(img_path)
            deleted_count += 1
        else:
            kept_count += 1

    # 3. Print Results
    print("\n=== Pruning Summary ===")
    if DRY_RUN:
        print("⚠️ DRY RUN MODE: No files were actually deleted.")
        print(f"Would keep:   {kept_count} images.")
        print(f"Would delete: {deleted_count} images.")
        print("If these numbers look correct, change DRY_RUN = False and run again.")
    else:
        print("✅ PRUNING COMPLETE")
        print(f"Kept:    {kept_count} images.")
        print(f"Deleted: {deleted_count} images.")


if __name__ == "__main__":
    prune_unused_coco_images()
