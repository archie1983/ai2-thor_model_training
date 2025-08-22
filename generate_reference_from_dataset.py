import tarfile
import os
import glob
import argparse


def main(args):
    tar_files = glob.glob(f"{args.data_path}/*.tar")
    dest_dir = args.dest_path

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Create Dir: {dest_dir}")

    total_extracted = 0

    for tar_path in tar_files:
        if not os.path.exists(tar_path):
            print(f"Warn: File {tar_path} not exist")
            continue

        try:
            with tarfile.open(tar_path, 'r') as tar:
                print(f"Now: {tar_path}")

                png_next_files = [
                    member for member in tar.getmembers()
                    if member.name.endswith('.png.next') and member.isfile()
                ]

                if not png_next_files:
                    print(f"There are no .png.next files")
                    continue

                for member in png_next_files:
                    tar.extract(member, path=dest_dir)
                    original_path = os.path.join(dest_dir, member.name)
                    new_path = original_path[:-5]
                    os.rename(original_path, new_path)
                    total_extracted += 1

                print(f"Extracted {len(png_next_files)} files")

        except Exception as e:
            print(f"Error while process in {tar_path} and the error is: {str(e)}")

    print(f"\nSuccess, files had been stored in {dest_dir}")
    print(f"Total extracted num: {total_extracted}")
    return total_extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dest-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
