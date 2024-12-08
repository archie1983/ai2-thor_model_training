from scene_data_management import SceneManagement
import pickle, os
from ai2_thor_utils import AI2THORUtils
from scene_description import SceneDescription
from ae_yolo_extractor import YOLOExtractor, YOLOType

from llm_room_classifier import LLMRoomClassifier # LLM room classifier
from room_type import RoomType
from ae_llm import LLMType
from time import time

class DataSceneProcessor:
    ##
    # We can override default data storage directory (normally- the name of LLM within experiment_data folder)
    ##
    def __init__(self, yolo_type, llm_type, data_store_dir = ""):
        # If we want to use YOLO to extrac objects, then create a yolo extractor
        if yolo_type != None:
            self.item_extractor = YOLOExtractor(yolo_type)

        self.NUMBER_OF_SCENES_IN_BATCH = 25
        self.atu = AI2THORUtils()
        self.yolo_type = yolo_type

        ##
        # We'll read pkl files from here - the ones that already have SVC or LLM classification stored
        ##
        if (data_store_dir == ""):
            self.data_store_dir = "experiment_data/" + "pkl_" + self.yolo_type.name
        else:
            self.data_store_dir = "experiment_data/" + data_store_dir

        # Storing copy of LLM type
        if llm_type != None:
            self.llm_type = llm_type
            # If we want to use LLM to classify rooms by their objects (extracted with YOLO),
            # then creating an LLM classifier
            self.lrc = LLMRoomClassifier(llm_type) # LLM classifier
            ## This is where we'll store new prepared pkl files - the ones that will also have YOLO extracted items stored
            self.data_store_dir_yolo = "experiment_data/" + "pkl_yolo_" + self.llm_type.name
        else:
            self.llm_type = None
            ## This is where we'll store new prepared pkl files - the ones that will also have YOLO extracted items stored
            self.data_store_dir_yolo = "experiment_data/" + "pkl_yolo_" + self.yolo_type.name

        self.scene_mgmt = SceneManagement(self.data_store_dir)

        self.DEBUG = False # A flag of whether we want to debug and go through scenes quickly - only analyzing some points.

    ##
    # Loads a single specified scene file from the data directory
    ##
    def load_scene_file(self, scene_id):
        scene_f = self.data_store_dir + "/scene_descr_" + str(scene_id) + ".pkl"
        ##
        # Check this scene file. If it doesn't exist, then return None.
        # If it does, then do something with it.
        ##
        try:
            f = open(scene_f,'rb')
        except FileNotFoundError:
            return None

        scene = pickle.load(f)

        return scene

    def store_scene_file_yolo(self, scene_id, sd):
        # store our room points collection into a pickle file
        # Create the directory where to store experiment data if it doesn't exist
        if not os.path.exists(self.data_store_dir_yolo):
            os.makedirs(self.data_store_dir_yolo)

        scene_descr_fname = self.data_store_dir_yolo + "/scene_descr_" + str(scene_id) + ".pkl"
        pickle.dump(sd, open(scene_descr_fname, "wb"))

    ##
    # Process a whole batch of scene files and don't stop until a batch size
    # has been processed.
    ##
    def process_1_batch_of_data_scenes(self):
        # scene descriptions. Each of which will contain points of its floorplan
        # that were traversed using the proper_convert_scene_to_grid_map_and_poses
        # method
        # If pickle file for our scene description exists, then move on to the next
        # scene, otherwise explore this one
        highest_scene_index = self.scene_mgmt.last_index_processed()
        print("Highest index processed: " + str(highest_scene_index))
        processed_scenes_in_this_batch = 0

        while processed_scenes_in_this_batch < self.NUMBER_OF_SCENES_IN_BATCH:
            scene_id = "train_" + str(highest_scene_index + 1)
            print("Processing " + scene_id)
            sd = self.load_scene_file(scene_id)
            highest_scene_index += 1
            if not sd:
                continue

            self.process_scene(sd, scene_id)

            processed_scenes_in_this_batch += 1

    ##
    # Processes a single scene that's already loaded from the data directory.
    ##
    def process_scene(self, scene_data, scene_id):
        points_cnt = 0

        new_sd_with_cvm = SceneDescription() # scene description - as before but now also classified with CVM

        room_points = scene_data.get_all_points()
        for point in room_points:
            # printing out a few already existing data about each point
            img_url = point["front_view_at_this_point"]
            print("################## NEW POINT:################# Image: " + img_url)
            objs_at_this_pos = self.atu.get_visible_object_names_from_collection_csv_unique(point["visible_objects_at_this_point"])
            print("Objects (AI2-THOR): " + objs_at_this_pos)
            print("Room Type GT: " + point["room_type_gt"].name)
            print("Room Type SVC: " + point["room_type_svc"].name)
            points_cnt += 1


            if self.llm_type != None:
                ## Now let's classify a room based on YOLO detected objects
                t0 = time()
                (rt_llm, full_ans_llm) = self.lrc.classify_room_by_this_object_set(objs_at_this_pos)
                llm_elapsed_time = round(time() - t0, 5)

                print("Room by YOLO + LLM: " + rt_llm.name)

                new_sd_with_cvm.addPoint(point["point_pose"],
                                        point["room_type_llm"],
                                        point["room_type_svc"],
                                        point["room_type_cvm"],
                                        rt_llm,
                                        point["room_type_gt"],
                                        point["visible_objects_at_this_point"],
                                        point["visible_objects_by_cvm"],
                                        point["items_in_imgage_yolo"],
                                        point["front_view_at_this_point"],
                                        point["elapsed_time_llm"],
                                        point["elapsed_time_svc"],
                                        point["elapsed_time_cvm"],
                                        llm_elapsed_time,
                                        point["llm_text"],
                                        point["cvm_text"],
                                        full_ans_llm
                                        )

            else:
                # Now let's try to analyze the pictures with YOLO and see what items do we see in each of them.
                items_in_imgage_yolo = self.item_extractor.what_is_in_the_picture(img_url)
                print("Items by YOLO: " + str(items_in_imgage_yolo))

                #items_in_image = self.crc.extract_items_from_this_image(img_url)
                #print("Items in image: " + items_in_image)
                #print("\n")

                new_sd_with_cvm.addPoint(point["point_pose"],
                                        point["room_type_llm"],
                                        point["room_type_svc"],
                                        point["room_type_cvm"],
                                        "",
                                        point["room_type_gt"],
                                        point["visible_objects_at_this_point"],
                                        point["visible_objects_by_cvm"],
                                        items_in_imgage_yolo,
                                        point["front_view_at_this_point"],
                                        point["elapsed_time_llm"],
                                        point["elapsed_time_svc"],
                                        point["elapsed_time_cvm"],
                                        0,
                                        point["llm_text"],
                                        point["cvm_text"],
                                        "")

            if self.DEBUG and points_cnt >= 3:
                break

        self.store_scene_file_yolo(scene_id, new_sd_with_cvm)

if __name__ == "__main__":
    ## To extract visual objects using YOLO World
    #dsp = DataSceneProcessor(YOLOType.WORLD, None, "pkl_CHAMELEON_full_prompt")
    #dsp.process_1_batch_of_data_scenes()

    ## To classify rooms using YOLO-World extracted visual objects
    dsp = DataSceneProcessor(None, LLMType.LLAMA, "pkl_yolo_WORLD")
    dsp.process_1_batch_of_data_scenes()
