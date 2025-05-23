from model_training import action_to_index, index_to_action
import numpy as np
from numpy.ma.core import argmax

##
# Instead of using CNN outputs directly, we may want to smooth the decisions and introduce
# some hysteresis. This class will help with that by treating the CNN softmax layer outputs
# as Fuzzy logic membership function outputs.
##
class FuzzyNavigationController:
    def __init__(self, memory_length=5):
        self.action_history = []
        self.decisions = []
        self.memory_length = memory_length

        self.left_action_ndx = action_to_index("RotateLeft")
        self.right_action_ndx = action_to_index("RotateRight")
        self.stop_action_ndx = action_to_index("STOP")
        self.ahead_action_ndx = action_to_index("MoveAhead")

    ##
    # Maintain memory of previous CNN outputs and smooth the decision.
    ##
    def get_smooth_action(self, cnn_outputs_in, hyster = False):
        #print("IN: ", cnn_outputs.squeeze())
        cnn_outputs = cnn_outputs_in.squeeze()

        # return values
        ret_action = "MoveAhead"
        ret_action_id = action_to_index("MoveAhead")
        ret_tensor = cnn_outputs

        # Store current raw prediction
        self.action_history.append(cnn_outputs)
        if len(self.action_history) > self.memory_length:
            self.action_history.pop(0)

        # Apply fuzzy rules for temporal smoothing
        smoothed_outputs = self.smooth_with_fuzzy_rules()

        raw_cnn_best_action_id = argmax(cnn_outputs_in)
        if (raw_cnn_best_action_id == self.stop_action_ndx or raw_cnn_best_action_id == self.ahead_action_ndx):
            '''
            If we only have STOP or MoveAhead then do that with no smoothing
            '''
            ret_action_id = raw_cnn_best_action_id
            ret_action = index_to_action(ret_action_id)
            ret_tensor = cnn_outputs
        elif hyster: # If we also want hysteresis, then continue, otherwise return what we have
            ret_action_id = self.make_decision_with_hysteresis(smoothed_outputs)
            ret_action = index_to_action(ret_action_id.item())
            ret_tensor = smoothed_outputs
        else:
            ret_action_id = argmax(smoothed_outputs)
            ret_action = index_to_action(ret_action_id)
            ret_tensor = smoothed_outputs

        return (ret_action, ret_action_id, ret_tensor)

    ##
    # Detect oscillation in current and previous states and if detected, then smooth the output
    ##
    def smooth_with_fuzzy_rules(self):
        if len(self.action_history) < 2:
            return self.action_history[-1]  # Not enough history

        current = self.action_history[-1]
        previous = self.action_history[-2]
        #print("CUR: ", current, " PREV: ", previous)

        # Detect oscillation (turn left -> turn right or vice versa)
        left_idx, right_idx = self.left_action_ndx, self.right_action_ndx # action indexes

        # normalize:
        current -= min(current.clone())
        current /= sum(current.clone())

        previous -= min(previous.clone())
        previous /= sum(previous.clone())

        # Maybe use max instead to measure oscillation strength?
        # Maybe we want to normalize the tensor first?
        # And maybe we want to only do it if we detect oscillation?
        oscillation_strength = max(
            previous[left_idx] * current[right_idx],
            previous[right_idx] * current[left_idx]
        )

        # Apply fuzzy rule: "IF strong oscillation THEN dampen response"
        damping_factor = 0.7  # Tunable parameter

        smoothed = current.clone()

        prev_action_id = argmax(previous)
        cur_action_id = argmax(current)
        if ((prev_action_id == left_idx and cur_action_id == right_idx)
                or (prev_action_id == right_idx and cur_action_id == left_idx)):

            if oscillation_strength > 0.3 or True:  # Fuzzy threshold
                print("Prev: ", previous, "os: ", oscillation_strength)
                print("Cur: ", current)
                # Apply temporal averaging with more weight on previous stable decisions
                for i in range(len(smoothed)):
                    # Weighted average of recent predictions
                    smoothed[i] = damping_factor * previous[i] + (1 - damping_factor) * current[i]

                print("Smo: ", smoothed)

        return smoothed

    ##
    # Introduce hysteresis in the smoothed outputs
    ##
    def make_decision_with_hysteresis(self, smoothed_outputs):
        # Find highest output
        max_idx = np.argmax(smoothed_outputs)
        #print("SO: ", smoothed_outputs, " MAXIDX: ", max_idx)

        # Check if we're considering turning and previously made a different turn
        if max_idx in [self.left_action_ndx, self.right_action_ndx]:  # Left or right turn
            # Get previous actual decision (not just output)
            if len(self.decisions) > 0:
                prev_decision = self.decisions[-1]

                # If switching between left and right, require stronger confidence
                if ((max_idx == self.left_action_ndx and prev_decision == self.right_action_ndx)
                        or (max_idx == self.right_action_ndx and prev_decision == self.left_action_ndx)):
                    # Require 20% more confidence to switch direction
                    second_best = np.argsort(smoothed_outputs)[-2]
                    if smoothed_outputs[max_idx] < smoothed_outputs[second_best] * 1.2:
                        # Not confident enough to switch, maintain previous direction
                        max_idx = prev_decision

        # Store this decision
        self.decisions.append(max_idx)
        return max_idx