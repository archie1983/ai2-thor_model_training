from enum import Enum
import glob, os, shutil, pickle
from . import DataLoadError
from pathlib import Path

##
# NavigationTrainingDataManagement class. It will deal with managing data files
# that are relevant to a particular habitat and training data extraction from it.
##
class NavigationTrainingDataManagement():
    def __init__(self, data_store_dir = "", collect_yolo_data = False, train_val_test = "train"):
        self.data_store_dir = data_store_dir
        self.collect_yolo_data = collect_yolo_data
        self.yolo_data_dir = self.data_store_dir + "/yolo"
        self.staging_dir_name = "staging"
        self.data_dir_prefix = "/h_"
        self.pkl_file_prefix = "/hm_"
        self.current_exploration_dir = ""
        self.exploration_id = 0
        self.current_metrics_store_fname = "" # this will be the file name for the pkl file
        self.expl_metrics_data = [] # we'll store metrics here in tuples for each exploration and each image
        self.habitat_metrics_data = [] # this will be metrics data at habitat level, so really a collection of expl_metrics_data collections

        # Create the directory where to store experiment data if it doesn't exist
        self.current_staging_dir = self.data_store_dir + self.data_dir_prefix + self.staging_dir_name

        # It shouldn't exist. If it does, then that's because of a failed run, delete it.
        if os.path.exists(self.current_staging_dir):
            shutil.rmtree(self.current_staging_dir)

        if self.collect_yolo_data:
            # check that we're collection train, test or val data
            if not train_val_test in ["train", "val", "test"]:
                raise DataLoadError(
                    "When collecting data for YOLO finetune, it has to be one of: 'train', 'val' or 'test' data.")

            # Create the directory where to store YOLO finetuning data if it doesn't exist.
            root = Path(self.yolo_data_dir)
            output_dirs = {
                "images_tr": root / "images" / "train",
                "images_v": root / "images" / "val",
                "images_te": root / "images" / "test",
                "labels_tr": root / "labels" / "train",
                "labels_v": root / "labels" / "val",
                "labels_te": root / "labels" / "test",
            }
            for d in output_dirs.values():
                d.mkdir(parents=True, exist_ok=True)

            self.yolo_img_dir = root / "images" / train_val_test
            self.yolo_lbl_dir = root / "labels" / train_val_test


    # Prepare for a new habitat exploration
    def start_habitat(self, habitat_id):
        self.habitat_id = habitat_id

        self.current_final_habitat_dir = self.data_store_dir + self.data_dir_prefix + str(self.habitat_id)

        # Create the directory where to store experiment data if it doesn't exist. In fact it shouldn't exist at this point
        if not os.path.exists(self.current_staging_dir):
            os.makedirs(self.current_staging_dir)
        else:
            raise DataLoadError("Staging directory exists when not expected. Are you running more than one instance of exploration?")

        # Prepare to store data in a new pkl file for this habitat
        self.current_metrics_store_fname = self.data_store_dir + self.pkl_file_prefix + str(self.habitat_id) + ".pkl"
        self.expl_metrics_data = []
        self.habitat_metrics_data = []

    ##
    # Adds the new exploration metrics to the data collection
    ##
    def add_metrics(self, new_metrics):
        self.expl_metrics_data.append(new_metrics)

    # End the current habitat exploration
    def end_habitat(self):
        # Rename the staging directory to the habitat directory
        # from ".../datastoredir/h_staging" to ".../datastoredir/h_x"
        if not os.path.exists(self.current_final_habitat_dir):
            os.rename(self.current_staging_dir, self.current_final_habitat_dir)

        # store our data collection into a pickle file
        pickle.dump(self.habitat_metrics_data, open(self.current_metrics_store_fname, "wb"))

        if self.collect_yolo_data:
            #self.yolo_img_dir = root / "images" / train_val_test
            #self.yolo_lbl_dir = root / "labels" / train_val_test
            # print("===============================================================================================")
            # print(self.habitat_metrics_data)
            # print("===============================================================================================")
            for exploration in self.habitat_metrics_data:
                expl_len, expl_data = exploration
                for expl_step in expl_data:
                    pos, act, remaining_len, all_img_data = expl_step
                    for img_data in all_img_data:
                        img_path, yolo_annotations = img_data
                        if len(yolo_annotations) > 0:
                            new_img_name = self.data_dir_prefix + str(self.habitat_id) + "_" + img_path.replace("/", "_")
                            copy_cmd = "cp " + self.current_final_habitat_dir + "/" + img_path + " " + str(self.yolo_img_dir) + new_img_name
                            os.popen(copy_cmd)
                            #print(copy_cmd)
                            label_path = str(self.yolo_lbl_dir) + new_img_name.replace(".png", ".txt")
                            with open(label_path, "w") as f:
                                f.write("\n".join(yolo_annotations))


    # Starts a new exploration in the current habitat
    def start_new_exploration(self):
        last_expl_index = self.last_processed_exploration_in_current_staging_dir()
        self.exploration_id = last_expl_index + 1

        self.current_exploration_dir = self.current_staging_dir + "/expl_" + str(self.exploration_id)
        if not os.path.exists(self.current_exploration_dir):
            os.makedirs(self.current_exploration_dir)

        self.expl_metrics_data = []

        return self.current_exploration_dir

    def end_current_exploration(self):
        self.habitat_metrics_data.append((len(self.expl_metrics_data), self.expl_metrics_data))

    ##
    # Return the file URI for the top view that we might want to store
    ##
    def get_current_top_view_fname(self):
        return self.current_staging_dir + "/top_expl_" + str(self.exploration_id)

    ##
    # Extract last exploration number from the current staging directory.
    # It looks at ".../datastoredir/h_staging/expl_x"
    ##
    def last_processed_exploration_in_current_staging_dir(self):
        if len(self.current_staging_dir) < len(self.data_dir_prefix + self.staging_dir_name):
            raise DataLoadError("Cannot extract last exploration number. Malformed staging directory.")

        return self.last_index_extracted(self.current_staging_dir + "/expl_*")

    ##
    # Extract last explored habitat ID. It looks at ".../datastoredir/h_train_x"
    ##
    def last_extracted_habitat(self):
        if len(self.data_store_dir) < 1:
            raise DataLoadError("Cannot extract last habitat number. Malformed data storage directory.")

        return self.last_index_extracted(self.data_store_dir + self.data_dir_prefix + "*")

    ##
    # Returns the highest index of explored habitats. It looks for a pattern of
    # "dir/dir/dir_number" and it will then extract the number from all such patterns
    # and return the highest number that it found. It can be used at different levels.
    # It will process the furthermost right number.
    ##
    def last_index_extracted(self, h_files_glob = ""):
        if (h_files_glob == ""):
            h_files_glob = self.data_store_dir + self.data_dir_prefix + "*"

        h_folders = glob.glob(h_files_glob) # habitats' folders
        #h_folders = [ name for name in os.listdir(h_files_glob) if os.path.isdir(os.path.join(h_files_glob, name)) ]

        highest_index = 0
        cur_index = 0

        for file_name in h_folders:
            els = file_name.split("_")
            cur_index = int(els[-1])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index
