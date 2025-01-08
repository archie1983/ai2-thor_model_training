from navigation_training_data_extractor import NavigationTrainingDataExtractor
from ae_robot_simulation_control import RobotNavigationControl

spp = NavigationTrainingDataExtractor("train_55")
rnc = RobotNavigationControl()
rnc.set_controller(spp.get_controller())
doors = spp.find_all_doors()

door_of_interest = doors[3]
print(door_of_interest)
path_and_plan = spp.get_path_to_actual_object(door_of_interest, is_door=True)
path = path_and_plan[0]
plan = path_and_plan[1]
spp.visualise_path(path)
