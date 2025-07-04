import os
import shutil
import tarfile


class DataTransformer:

    def __init__(self, dir):
        self.dir = dir

    def rename(self, new_dir):
        print(len(os.listdir(self.dir)))
        os.makedirs(new_dir, exist_ok=True)

        for folder_name in os.listdir(self.dir):
            # Skip, only folder is what we need
            if not os.path.isdir(os.path.join(self.dir, folder_name)):
                continue

            # Rename and move to new dir
            print('Folder name: ', folder_name)
            folder_path = os.path.join(self.dir, folder_name)
            for filename in os.listdir(folder_path):
                original_path = os.path.join(folder_path, filename)
                new_filepath = os.path.join(new_dir, folder_name + '_' + filename)
                shutil.copy2(original_path, new_filepath)


    def trans_to_web_dataset(self, store_dir, output_dir):
        print(len(os.listdir(store_dir)))

        with tarfile.open(output_dir, "w") as tar:
            for filename in os.listdir(store_dir):
                if filename.endswith('.png'):
                    prefix = filename[:-4]
                    img_path = os.path.join(store_dir, f"{prefix}.png")
                    json_path = os.path.join(store_dir, f"{prefix}.json")
                    tar.add(img_path, arcname=f"{prefix}.png")
                    tar.add(json_path, arcname=f"{prefix}.json")

        print(f"Is over, output dir: {output_dir}")


if __name__ == '__main__':
    store_path = os.path.join(os.getcwd(), 'harvested_data/h_1')
    new_dir = os.path.join(os.getcwd(), 'dataset/train')
    output_dir = os.path.join(os.getcwd(), 'dataset/tar')

    #data_transformer = DataTransformer(store_path)
    #data_transformer.trans_to_web_dataset(new_dir, output_dir)
