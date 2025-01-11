from enum import Enum
import glob, os

class ClassificationMethod(Enum):
    SVC = 1
    LLM = 2
    CVM = 3
    SVC_LLM = 4
    SVC_CVM = 5
    SVC_CVM_LLM = 6

    #@classmethod
    def svc_required(self):
        if self == ClassificationMethod.SVC or self == ClassificationMethod.SVC_LLM or self == ClassificationMethod.SVC_CVM  or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

    #@classmethod
    def llm_required(self):
        if self == ClassificationMethod.LLM or self == ClassificationMethod.SVC_LLM or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

    #@classmethod
    def cvm_required(self):
        if self == ClassificationMethod.CVM or self == ClassificationMethod.SVC_CVM  or self == ClassificationMethod.SVC_CVM_LLM:
            return True
        else:
            return False

class SceneManagement():
    def __init__(self, data_store_dir):
        self.data_store_dir = data_store_dir

    ##
    # Returns the highest index of scenes explored
    ##
    def last_index_extracted(self, pkl_files_glob = ""):
        if (pkl_files_glob == ""):
            pkl_files_glob = self.data_store_dir + "/scene_descr_train_*.pkl"

        scene_files = glob.glob(pkl_files_glob) # scene files

        highest_index = 0
        cur_index = 0

        for file_name in scene_files:
            els = file_name.split("_")
            cur_index = int(els[-1][:-4])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index

    ##
    # Returns the highest index of scenes processed by the DataSceneProcessor
    ##
    def last_index_processed(self, pkl_files_glob = ""):
        if (pkl_files_glob == ""):
            pkl_files_glob = self.data_store_dir + "/scene_results_train_*.pkl"

        scene_files = glob.glob(pkl_files_glob) # scene files

        highest_index = 0
        cur_index = 0

        for file_name in scene_files:
            els = file_name.split("_")
            cur_index = int(els[-1][:-4])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index

class NavigationTrainingDataManagement():
    def __init__(self, data_store_dir):
        self.data_store_dir = data_store_dir
        self.staging_dir = "staging"
        self.current_staging_dir = ""
        self.current_exploration_dir = ""

    # Prepare for a new habitat exploration
    def start_habitat(self, habitat_id):
        self.habitat_id = habitat_id
        # Create the directory where to store experiment data if it doesn't exist
        self.current_staging_dir = self.data_store_dir + "/h_" + self.staging_dir
        if not os.path.exists(self.current_staging_dir):
            os.makedirs(self.current_staging_dir)

    # End the current habitat exploration
    def end_habitat(self):
        # Rename the staging directory to the habitat directory
        if not os.path.exists(self.data_store_dir + "/h_" + self.habitat_id):
            os.rename(self.data_store_dir + "/" + self.staging_dir, self.data_store_dir + "/" + self.habitat_id)

    # Starts a new exploration in the current habitat
    def start_new_exploration(self):
        last_expl_index = self.last_index_extracted(self.data_store_dir + "/h_" + self.staging_dir + "/expl_*")
        exploration_id = last_expl_index + 1

        self.current_exploration_dir = self.data_store_dir + "/h_" + self.staging_dir + "/expl_" + exploration_id
        if not os.path.exists(self.current_exploration_dir):
            os.makedirs(self.current_exploration_dir)

        return self.current_exploration_dir

    ##
    # Returns the highest index of explored habitats
    ##
    def last_index_extracted(self, h_files_glob = ""):
        if (h_files_glob == ""):
            h_files_glob = self.data_store_dir + "/h_*"

        h_folders = glob.glob(h_files_glob) # habitats' folders

        highest_index = 0
        cur_index = 0

        for file_name in h_folders:
            els = file_name.split("_")
            cur_index = int(els[-1])
            if (cur_index > highest_index):
                highest_index = cur_index

        return highest_index
