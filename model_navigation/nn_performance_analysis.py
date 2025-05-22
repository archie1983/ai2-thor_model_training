from . import SceneAnalyzer, FuzzyNavigationController
import pickle
import numpy as np

##
# This class will use one or more of our neural network models and navigate through a scene
##
class NNPerformanceAnalyzer():
    def __init__(self, pth_path = "accuracy_093.pth"):
        # Load a CNN that tells us the next best move
        self.sa = SceneAnalyzer(pth_path)
        self.data_dir = ""

        # Fuzzifier and hysteresis machine for our CNN decisions
        self.fnc = FuzzyNavigationController()

    def process_required_habitats(self):
        # To keep track of total errors
        total_errors = None
        total_correct = None
        gt = None

        scenes_to_evaluate = [501, 502, 503, 504, 505, 610, 506, 515, 510]
        #scenes_to_evaluate = [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]

        for scene_id in scenes_to_evaluate:
            (habitat_errors, habitat_correct, habitat_gt) = self.process_training_scenes_in_habitat(scene_id)
            total_errors = self.add_dicts(habitat_errors, total_errors)
            total_correct = self.add_dicts(habitat_correct, total_correct)
            gt = self.add_dicts(habitat_gt, gt)

        error_rate = {
            key: (gt[key] - total_correct[key]) / gt[key] * 100
            for key in gt
            if gt[key] != 0  # Avoid division by zero
        }

        print("Error rate: ", error_rate)

    ##
    # Here we will go through the scenes that were used in training and verify
    # how well do the predicted actions match the actual best action at each step.
    ##
    def process_training_scenes_in_habitat(self, habitat_id):
        # First open the habitat pickle file
        self.data_dir = "harvested_data/h_" + str(habitat_id) + "/"
        hab_file = "harvested_data/hm_" + str(habitat_id) + ".pkl"
        f = open(hab_file, 'rb')
        hab_data = pickle.load(f)

        # To keep track of habitat errors
        habitat_errors = None
        habitat_correct = None
        habitat_gt = None

        # Here we have a lot of trajectories. Each trajectory will have a starting point. It's that point
        # that we want to teleport to and then choose the next best action and then compare with the ground truth
        # action.
        for hab in hab_data:
            # Habitat data contains a tuple of step count in the path and the navigation sequence (a list of tuples)
            (step_cnt, nav_seq) = hab
            # Each member in navigation sequence is a tuple of variables at that step:
            # pose, next action, remaining path length and img_uris of view around agent.
            #(start_pose, next_action, path_length, img_uris) = nav_seq[0]

            ## Let's teleport to the start position
            #self.rnc.teleport_to(start_pose)

            #print(self.process_nav_sequence(nav_seq))
            sequence_results = self.process_nav_sequence(nav_seq)

            # calculate errors in this navigation sequence
            sequence_errors = self.subtract_dicts(sequence_results["gt_stats"], sequence_results["correctness_stats"])
            #print(sequence_errors)

            # Add the errors of the sequence to the total habitat errors
            habitat_errors = self.add_dicts(sequence_errors, habitat_errors)

            # Add the correct inferences in this habitat
            habitat_correct = self.add_dicts(sequence_results["correctness_stats"], habitat_correct)

            # Count ground truth stats
            habitat_gt = self.add_dicts(sequence_results["gt_stats"], habitat_gt)

        #print("Habitat Errors: ", habitat_errors)
        return (habitat_errors, habitat_correct, habitat_gt)

    ##
    # Subtracts 2 dictionaries key by key. Useful for analysing errors.
    ##
    def subtract_dicts(self, dict1, dict2):
        if dict1 is None and dict2 is not None:
            res_dict = dict2
        if dict2 is None and dict1 is not None:
            res_dict = dict1
        else:
            res_dict = {
                key: dict1[key] - dict2[key]
                for key in dict1
            }
        return res_dict

    def add_dicts(self, dict1, dict2):
        if dict1 is None and dict2 is not None:
            res_dict = dict2
        if dict2 is None and dict1 is not None:
            res_dict = dict1
        else:
            res_dict = {
                key: dict1[key] + dict2[key]
                for key in dict1
            }
        return res_dict

    ##
    # Use a neural network to infer the next best action and compare to ground truth.
    ##
    def process_nav_sequence(self, nav_seq):
        # Statistics about ground truth choices
        ground_truth_move_stats = {
            "RotateLeft": 0,
            "RotateRight": 0,
            "MoveAhead": 0,
            "STOP": 0
        }

        # Statistics about inferred choices
        inferred_move_stats = {
            "RotateLeft": 0,
            "RotateRight": 0,
            "MoveAhead": 0,
            "STOP": 0
        }

        # Statistics about how many inferred choices matched ground truth
        correct_inference_stats = {
            "RotateLeft": 0,
            "RotateRight": 0,
            "MoveAhead": 0,
            "STOP": 0
        }

        # Process all navigation elements in navigation sequence.
        for nav_el in nav_seq:
            (cur_pose_gt, next_action_gt, path_length_gt, img_uris) = nav_el
            pic_of_interest = self.data_dir + img_uris[0]
            next_move_str, next_move_index, softmax = self.sa.next_best_move(scene_img_url=pic_of_interest)
            #next_move_str, next_move_index, softmax = self.fnc.get_smooth_action(softmax, False)
            #next_move_str, next_move_index, softmax = self.fnc.get_smooth_action(softmax, True)

            # gather statistics about ground truth
            ground_truth_move_stats[next_action_gt] += 1
            # gather statistics about inference choice
            inferred_move_stats[next_move_str] += 1
            # gather statistics about how inference choice matches ground truth
            if (next_action_gt == next_move_str):
                correct_inference_stats[next_action_gt] += 1

        # Now return the stats
        return {
                "inference_cnt": sum(ground_truth_move_stats.values()),
                "gt_stats": ground_truth_move_stats,
                "inference_stats": inferred_move_stats,
                "correctness_stats": correct_inference_stats
        }

if __name__ == "__main__":
    npa = NNPerformanceAnalyzer()
    npa.process_training_scenes_in_habitat(10)