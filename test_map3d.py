# Copyright 2022 Kaiyu Zheng
# 
# Usage of this file is licensed under the MIT License.

import time
import copy
from PIL import Image
import thortils as tt
from thortils import constants
from thortils.controller import launch_controller, thor_controller_param
from thortils.map3d import Map3D, Mapper3D
from thortils.utils.visual import GridMapVisualizer, Visualizer2D
from thortils.agent import thor_reachable_positions
from thortils.scene import proper_convert_scene_to_grid_map
from thortils.vision import thor_topdown_img

def test_mapper_top_down(scene, floor_cut=0.1):
    controller = launch_controller({**constants.CONFIG, **{'scene': scene}})
    # grid_map = proper_convert_scene_to_grid_map(controller, floor_cut=floor_cut)


    # topdown_img = thor_topdown_img(controller)
    # img = Image.fromarray(topdown_img)
    # img.show()
    
    mapper = Mapper3D(controller)

    mapper.automate(num_stops=20, sep=1.5)
    grid_map = mapper.get_grid_map(floor_cut=floor_cut, debug=False)

    # Visualize reachable positions obtained from controller
    reachable_positions = thor_reachable_positions(controller)
    highlights = []
    for thor_pos in reachable_positions:
        highlights.append(grid_map.to_grid_pos(*thor_pos))
        # print("ROXXI: thor_pos: ", thor_pos)
    print("ROXXI: highlights: ", highlights)
    array = [(14,8)]
    # show grid map
    viz = GridMapVisualizer(grid_map=grid_map, res=30)
    img = viz.render()
    img = viz.highlight(img, highlights,
                        color=(244, 0, 224), show_progress=True)  # 紫色
    img = viz.highlight(img, array,
                        color=(0, 244, 224), show_progress=True)  # 蓝色
    viz.show_img(img)
    time.sleep(100)
    viz.on_cleanup()
    controller.stop()




if __name__ == "__main__":
    # test_mapper_top_down("FloorPlan2")
    test_mapper_top_down("FloorPlan1")
    # test_mapper_top_down("FloorPlan3")
    # test_mapper_top_down("FloorPlan4")
    # test_mapper_top_down("FloorPlan201", floor_cut=0.3)
    # test_mapper_top_down("FloorPlan202")
    # test_mapper_top_down("FloorPlan301")
    # test_mapper_top_down("FloorPlan302")
    # test_mapper_top_down("FloorPlan303")
    # test_mapper_top_down("FloorPlan401")
    # test_mapper_top_down("FloorPlan402")
    # test_mapper_top_down("FloorPlan403")
