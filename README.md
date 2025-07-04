The project relies on [AI2-THOR](https://github.com/allenai/ai2thor) and [ProcTHOR-10k](https://github.com/allenai/procthor-10k) dataset

Note: This branch is specifically intended for extracting data for the diffusion model. The data format consists of the robot's spatial information (x, y, z, orientation).

The entry point for code execution (data extraction) is located at:
training_data_extractor/diffusion_training_data_extractor.py

After extraction, the data needs to be structured into a dataset (formatted as WebDataset). Please use:
data_transform.py

Please download and install them before running this code. The AI2-THOR used is v5.0.0.

You must also install Thortils version from the referenced repository in the git submodule here. The vanilla Thortils will not work as I made several important changes. The best way to do that is to set up a conda environment for this purpose and install Thortils using:

```
git clone https://github.com/archie1983/ai2-thor_model_training
cd ai2-thor_model_training
git submodule init
git submodule update
cd thortils
pip install --no-cache-dir -e .
```

To demonstrate our approach, there are three main scripts- all very basic and can be copy-pasted into a jupyter notebook or run from terminal. The real code is deeper in the **training_data_extraction**, **model_training** and **model_navigation** packages.

**harvest_data.py** - for extracting training data from AI2-THOR and ProcTHOR-10k dataset. It will go through ProcTHOR-10k habitats, place the agent at random positions and walk from there to the room centre taking pictures along the way. It will also store metrics in pickle files. The pickle files contain remaining path length at each step, next action to be taken and the relative URI of the images. It currently takes 3 images- the front view and two side views at 120 degrees away from the front view. Currently it only walks to the centre of the room, not yet to any defined object. The collected data is stored in **harvested_data** directory.

**train_models.py** - Uses the data in the **harvested_data** directory, curates a custom PyTorch dataset, then creates a DataLoader, then sets up a neural network (currently just an insignificant CNN architecture without much thought) and feeds data into the neural network for training.

**navigate_scene.py** - Uses a model trained with the **model_training** package to navigate a given scene- currently only to navigate from a random location to the centre of the room.

This comprises a starter code for my MSc students. The **training_data_extraction** package needs to be extended to navigate to other objects, not just the centre of the room. The **model_training** package needs to be extended to train whatever the task requires- whether it's a classifier or a diffusion network or something else entirely.

Please use the environment.yml file to create the required conda environment to run this software in.

## Training Data Extraction Package: ##
* ae_robot_simulation_control.py : This is where we move robot around in simulation, execute action plans (planned paths) and take pictures at each step.
* navigation_training_data_extractor.py : This is where we go through all habitats, plan paths and then use ae_robot_simulation_control to walk through them.
* scene_data_management.py : This is where we start and stop habitat explorations. We also organise pkl files here.
* ai2_thor_utils.py : Various utility functions that we need in order to interact with AI2-THOR the way we do.
* room_type.py : An enum to differentiate room types. It has some more functionality than needed because it was borrowed from a different project. This could be cleaned up, but for now it doesn't bother anyone.
* data_load_error.py : An error that we can throw if data loading fails in one way or another.

## Model Training Package: ##
* habitat_dataset.py : This is where we go throuh all the generated pkl files and stored images and prepare training data out of them in a PyTorch set.
* habitat_data_loading.py : This is where take the PyTorch set from habitat_dataset.py and turn it into a PyTorch DataLoader object. We also apply an image transform- normalization really and stacking the images if we have more than one, but that may well change.
* habitat_neural_network.py : This is where we define our neural network architectures that we may want to experiment with. Also forward pass function lives here.
* habitat_nn_trainer.py : This is where we train and test our model and do epochs of training. We also have model saving and loading functions here.
* habitat_nn_utils.py : Utility functions, e.g., model loading code, action mapping, image transform, etc.
* hyper_params.py : This is a structure for hyper parameters. We can instantiate this class with various parameters to have different sets of hyper parameters.

## Model Navigation Package: ##
* fuzzy_navigation_controller.py : Various methods to smoothen NN navigation decisions to prevent oscillation and similar adversary effects.  
* nn_performance_analysis.py : Neural Network performance analysis. Currently containing some evaluation code for the navigation CNN.
* scene_analyzer.py : This is where we use a model to infer the next best action or score or other things related to a scene.
* scene_navigator.py : This is where we generate random placements in a scene and then use inference results from **secene_analyzer.py** 
to apply them to actual navigation in that scene from the random placements.
