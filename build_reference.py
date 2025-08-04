import tarfile
import os


def extract_png_from_tar(tar_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        png_members = [m for m in tar.getmembers() if m.name.endswith(".png")]

        print(f"Found {len(png_members)} PNG files in tar.")

        for member in png_members:
            tar.extract(member, path=output_dir)
            # Optionally move them all to output_dir root
            src_path = os.path.join(output_dir, member.name)
            dst_path = os.path.join(output_dir, os.path.basename(member.name))
            os.rename(src_path, dst_path)

        # Clean up any subdirectories if they were created
        for member in png_members:
            parent_dir = os.path.join(output_dir, os.path.dirname(member.name))
            if os.path.isdir(parent_dir) and parent_dir != output_dir:
                try:
                    os.rmdir(parent_dir)
                except OSError:
                    pass

    print(f"Extraction completed. PNGs are saved to: {output_dir}")


# ==================== Example Usage ====================
if __name__ == "__main__":
    tar_file = "D:\Dataset\Diffusion\\test1.tar"
    output_dir = os.path.join(os.getcwd(), "/out")
    extract_png_from_tar(tar_file, output_dir)
