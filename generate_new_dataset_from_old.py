import os
import json
import re
import tarfile
import random
import math
from PIL import Image
import io
from tqdm import tqdm


# Global
directions = [0, 1]


def resize_image(input_path, size=(256, 256)):
    """将图片resize到指定大小"""
    with Image.open(input_path) as img:
        img = img.resize(size, Image.LANCZOS)
        return img


def get_all_images(input_dir):
    angles = [45, 90, 135, 180, 225, 270, 315]
    # Pattern for matching base image name (expl_1_1.png)
    base_pattern = re.compile(r'expl_(\d+)_(\d+)\.png')
    # Data Structure: {base_room_name: [base, rotate1, rotate2 ... rotate7]}
    room_data = {}

    # Scan folder and classify
    for filename in os.listdir(input_dir):
        # Match base image name
        base_match = base_pattern.match(filename)
        if base_match:
            room_id = base_match.group(1)
            image_id = int(base_match.group(2))
            base_name = f"expl_{room_id}_{image_id}.png"
            rotated_image_names = [f"expl_{room_id}_{angle}_{image_id}.png" for angle in angles]
            room_data[base_name] = [base_name]
            for name in rotated_image_names:
                room_data[base_name].append(name)

    return room_data


def random_get_one_record(room_data):
    # Random get a base name
    base_name = random.choice(list(room_data.keys()))
    all_images = room_data[base_name]

    # Safe check
    # assert len(all_images) >= 2
    if len(all_images) < 2:
        return None

    # Random get a current image
    current_index = random.choice(range(len(all_images)))
    direction = random.choice(directions)
    # Get next image (and in case of Out of index)
    if direction == directions[0]:
        next_index = current_index - 1
        if next_index < 0:
            next_index = len(all_images) - 1
    else:
        next_index = current_index + 1
        if next_index >= len(all_images):
            next_index = 0

    # Json
    turn_json = {"direction": direction}

    # current_image_name, next_image_name, json
    return room_data[base_name][current_index], room_data[base_name][next_index], turn_json


def convert_dataset(input_dir, output_path, record_num, shard_num, target_size=(256, 256)):
    os.makedirs(output_path, exist_ok=True)
    room_data = get_all_images(input_dir)
    records_per_shard = math.ceil(record_num / shard_num)
    duplicate_set = set()
    counter = 1

    # Build dataset .tar
    for shard_id in range(shard_num):
        shard_filename = os.path.join(output_path, f"dataset-{shard_id:05d}.tar")
        with tarfile.open(shard_filename, 'w') as tar:
            for _ in tqdm(range(records_per_shard), desc=f"Shard {shard_id}", leave=False):
                if counter >= record_num:
                    break
                cur_name, next_name, turn_json = random_get_one_record(room_data)
                if cur_name + next_name in duplicate_set:
                    continue
                duplicate_set.add(cur_name + next_name)
                cur_img_path = os.path.join(input_dir, cur_name)
                next_img_path = os.path.join(input_dir, next_name)
                # New name in .tar
                new_base = f"{counter:05d}"
                current_png_name = f"{new_base}.png.cur"
                target_png_name = f"{new_base}.png.next"
                json_name = f"{new_base}.json"
                # Resize and add to .tar
                try:
                    # Current image
                    c_img = resize_image(cur_img_path, target_size)
                    c_bytes = io.BytesIO()
                    c_img.save(c_bytes, format='PNG')
                    c_bytes.seek(0)

                    info = tarfile.TarInfo(current_png_name)
                    info.size = len(c_bytes.getvalue())
                    tar.addfile(info, c_bytes)

                    # Next image
                    n_img = resize_image(next_img_path, target_size)
                    n_bytes = io.BytesIO()
                    n_img.save(n_bytes, format='PNG')
                    n_bytes.seek(0)

                    info = tarfile.TarInfo(target_png_name)
                    info.size = len(n_bytes.getvalue())
                    tar.addfile(info, n_bytes)

                except Exception as e:
                    print(f"Error of processing an image: {e}")
                    continue

                # Json to .tar
                json_bytes = io.BytesIO()
                json_bytes.write(json.dumps(turn_json).encode('utf-8'))
                json_bytes.seek(0)
                info = tarfile.TarInfo(json_name)
                info.size = len(json_bytes.getvalue())
                tar.addfile(info, json_bytes)
                counter += 1
        print(f"Shard {shard_id} finished: {shard_filename}")


if __name__ == "__main__":
    input_path = r"D:\Dataset\Diffusion\新建文件夹\test1"
    output_path = r"D:\Dataset\Diffusion\新建文件夹"
    record_num = 18968
    shard_num = 4

    convert_dataset(input_path, output_path, record_num, shard_num, target_size=(256, 256))
    print(f"Dataset had been converted and resize, saved to {output_path}")
    print("Each image:")
    print(f"[id].png.cur - cur img")
    print(f"[id].png.next - next img")
    print(f"[id].json - turn info")