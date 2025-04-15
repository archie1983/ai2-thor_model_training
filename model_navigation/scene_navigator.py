from . import SceneAnalyzer
from training_data_extraction import RobotNavigationControl
from thortils import launch_controller
from thortils.utils.math import sep_spatial_sample
import thortils as tt
import prior, random

##
# This class will use one or more of our neural network models and navigate through a scene
##
class SceneNavigator():
    def __init__(self):
        # Load a CNN that tells us the next best move
        #self.sa = SceneAnalyzer("accuracy_093.pth")
        self.rnc = RobotNavigationControl()
        self.dataset = None
        self.controller = None

    def process_required_habitats(self):
        self.process_habitat(10)
        self.controller.stop()

    ##
    # Process the given habitat- load it, put agent in random places and navigate from those places to some set goal.
    ##
    def process_habitat(self, habitat_id):
        # load required habitat
        habitat = self.load_proctor_habitat(habitat_id)

        # Launch a controller for the loaded habitat. If we already have a controller,
        # then reset it instead of loading a new one.
        if (self.controller == None):
            self.controller = launch_controller({"scene": habitat, "VISIBILITY_DISTANCE": 3.0, "headless": False})
            self.rnc.set_controller(self.controller) # This allows our control scripts to interact with AI2-THOR environment
        else:
            self.controller.reset(habitat)
            self.reset_state()
            self.rnc.reset_state()
            #self.rnc.set_controller(self.controller)

        self.process_placements_in_habitat()

    ##
    # Here we will select a number of random placements and then attempt to navigate from each of them
    # to some goal.
    ##
    def process_placements_in_habitat(self):
        ## All we need is a set of random positions and we get them like this:
        # params for the random teleportation part
        seed = 1983
        num_stops = 20
        num_rotates = 4
        sep = 1.0
        v_angles = [30]
        h_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        """
        num_stops: Number of places the agent will be placed
        num_rotates: Number of random rotations at each place
        sep: the minimum separation the sampled agent locations should have

        kwargs: See thortils.vision.projection.open3d_pcd_from_rgbd;
        """
        rnd = random.Random(seed)

        initial_agent_pose = tt.thor_agent_pose(self.controller)
        initial_horizon = tt.thor_camera_horizon(self.controller.last_event)

        reachable_positions = tt.thor_reachable_positions(self.controller)
        placements = sep_spatial_sample(reachable_positions, sep, num_stops,
                                        rnd=rnd)

        #print(placements)

        explorations_processed = 0
        for p in placements:
            # append a rotation to the place.
            yaw = rnd.sample(h_angles, 1)[0]
            place_with_rtn = p + (yaw,)
            print("Placement: ", place_with_rtn)
            ## Teleport, then start new exploration. Achieve goal. Then repeat.
            self.rnc.teleport_to(place_with_rtn)

            explorations_processed += 1

    ##
    # Get Procthor-10k dataset
    ##
    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            #print(self.dataset)
        return self.dataset

    ##
    # Load a PROCTHOR scene specified by the habitat_id.
    ##
    def load_proctor_habitat(self, habitat_id):
        dataset = self.getDataSet()
        self.habitat_id = habitat_id
        print("Loading : train[" + str(habitat_id) + "]")
        house = dataset["train"][habitat_id]
        return house

#if __name__ == "__main__":
#    sn = SceneNavigator()
#    sn.process_habitat(10)