import glob
import pickle

def process_scene_files():
    pkl_store = "harvested_data/*.pkl"
    hab_files = glob.glob(pkl_store) # files showing gemma scenes

    for hab_f in hab_files:
        f = open(hab_f,'rb')
        hab_data = pickle.load(f)
        print(hab_f)

        for i in range(len(hab_data)):
            print("Hab data: ", hab_data[i])

process_scene_files()
