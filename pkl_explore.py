import glob
import pickle
from scene_description import SceneDescription

def find_observed_point_by_pose(pose, room_points):
    #(pos, rot) = pose # ((10.75, 1.57599937915802, 1.0), (30.000003814697266, 0.0, 0))
    result = None
    for rp in room_points:
        if rp['point_pose'] == pose:
            result = rp
            return result
    return result

def process_scene_files():
    #pkl_store = "pkl_explore/scene_descr_train_*.pkl"
    #pkl_store = "experiment_data/pkl_CHAMELEON/*.pkl"
    pkl_store = "experiment_data/pkl_yolo_WORLD/*.pkl"
    #pkl_store = "experiment_data/pkl_MOONDREAM_one_word/*.pkl"
    #pkl_store = "experiment_data/pkl_LLAMA/scene_descr_train_10.pkl"
    #pkl_store = "experiment_data/pkl_yolo_MEDIUM/scene_descr_train_10.pkl"

    scene_files = glob.glob(pkl_store) # files showing gemma scenes
    #scene_files2 = glob.glob(pkl_store2)

    all_seen_objs = set()

    for scene_f in scene_files:
        f = open(scene_f,'rb')
        scene = pickle.load(f)

        f_name = scene_f.split("/")[2]
        cor_llm_pkl_path = "experiment_data/pkl_LLAMA/" + f_name # corresponding LLM pkl path
        llm_f = open(cor_llm_pkl_path,'rb')
        llm_scene = pickle.load(llm_f)

        llm_room_points = llm_scene.get_all_points()

        print(scene_f)
        #print(scene)

        room_points = scene.get_all_points()
        #print(len(room_points))
        #print(room_points[0])
        for i in range(len(room_points)):
            #print(room_points[i]['room_type_svc'].name + " :: " + room_points[i]['room_type_cvm'].name)
            #print(room_points[i].keys())
            print("YOLO: ", room_points[i]['visible_objects_by_yolo'])
            #print("AI2-THOR: ", room_points[i]['visible_object_names'])
            all_seen_objs.update(room_points[i]['visible_object_names'])
            #fp = find_observed_point_by_pose(room_points[i]['point_pose'], llm_room_points)
            #print(str(fp['point_pose']))

    print(len(all_seen_objs), all_seen_objs)
    obj_str = ""
    for item in all_seen_objs:
        obj_str += ", " + item

    obj_str = obj_str[1:]
    print(obj_str)

process_scene_files()
