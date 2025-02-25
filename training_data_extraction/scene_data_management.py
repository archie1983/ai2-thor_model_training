from enum import Enum
import glob, os, shutil
from . import DataLoadError

##
# NavigationTrainingDataManagement class. It will deal with managing data files
# that are relevant to a particular habitat and training data extraction from it.
##
class NavigationTrainingDataManagement():
    def __init__(self, data_store_dir):
        self.data_store_dir = data_store_dir
        self.staging_dir_name = "staging"
        self.data_dir_prefix = "/h_"
        self.current_exploration_dir = ""
        # Create the directory where to store experiment data if it doesn't exist
        self.current_staging_dir = self.data_store_dir + self.data_dir_prefix + self.staging_dir_name

        # It shouldn't exist. If it does, then that's because of a failed run, delete it.
        if os.path.exists(self.current_staging_dir):
            shutil.rmtree(self.current_staging_dir)

    # Prepare for a new habitat exploration
    def start_habitat(self, habitat_id):
        self.habitat_id = habitat_id

        self.current_final_habitat_dir = self.data_store_dir + self.data_dir_prefix + str(self.habitat_id)

        # Create the directory where to store experiment data if it doesn't exist. In fact it shouldn't exist at this point
        if not os.path.exists(self.current_staging_dir):
            os.makedirs(self.current_staging_dir)

    # End the current habitat exploration
    def end_habitat(self):
        # Rename the staging directory to the habitat directory
        # from ".../datastoredir/h_staging" to ".../datastoredir/h_x"
        if not os.path.exists(self.current_final_habitat_dir):
            os.rename(self.current_staging_dir, self.current_final_habitat_dir)

    # Starts a new exploration in the current habitat
    def start_new_exploration(self):
        last_expl_index = self.last_processed_exploration_in_current_staging_dir()
        exploration_id = last_expl_index + 1

        self.current_exploration_dir = self.current_staging_dir + "/expl_" + exploration_id
        if not os.path.exists(self.current_exploration_dir):
            os.makedirs(self.current_exploration_dir)

        return self.current_exploration_dir

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

        highest_index = 0
        cur_index = 0

        for file_name in h_folders:
            els = file_name.split("_")
            cur_index = int(els[-1])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index
