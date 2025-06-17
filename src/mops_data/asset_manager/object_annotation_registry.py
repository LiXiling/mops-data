import re

import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Actor, Link

from mops_data.asset_manager import anno_handler as mops_ah

# in some edgecases, the loaded robocasa object names do not conform to the class list
EDGE_CASE_MAP = {
    "LightSwitch": "Switch",
    "SoapDispenser": "Dispenser",
    "Cab": "Cabinet",
}


def _handle_edge_cases(class_name):
    # Normalize Edge cases
    if class_name in EDGE_CASE_MAP:
        class_name = EDGE_CASE_MAP[class_name]
    return class_name


def parse_robocasa_actor(obj_name):
    name_parts = obj_name.split("_")
    class_name = [name_parts[1]]

    for i in range(2, len(name_parts)):
        n = name_parts[i]
        if n.isnumeric():
            break
        if n in ["left", "right", "main", "room"]:
            break
        class_name.append(n)

    # capitalize each entry in class_name and concatenate them
    class_name = "".join([name.capitalize() for name in class_name])
    # remove -workspace if its in classname
    class_name = re.sub("-workspace", "", class_name)
    return class_name


def parse_robocasa_link(obj_name):
    class_name = obj_name.split("_")[0].split("-")[-1].capitalize()
    return class_name


def parse_name(obj):
    obj_name = obj._objs[0].name
    class_name = obj_name
    if isinstance(obj, Link):
        class_name = parse_robocasa_link(obj_name)
    if isinstance(obj, Actor):
        class_name = parse_robocasa_actor(obj_name)
    class_name = _handle_edge_cases(class_name)
    return class_name, obj.name.capitalize()


class ObjectAnnotationRegistry:
    def __init__(self):
        self.anno_handler: mops_ah.AnnotationHandler = mops_ah.load_annotations()

        self.affordance_id_map = self.anno_handler.affordance_id_map

        self.segm_id_to_class_id_map = {}
        self.segm_id_to_affordance_id_map = {}

    def get_num_affords(self):
        return len(self.affordance_id_map)

    def get_class_id(self, obj_id):
        return self.segm_id_to_class_id_map.get(obj_id, -1)

    def add_partnet_object(
        self,
        partnet_link_obj: Link,
        class_name: str,
        linkname_to_affordances: dict,
    ):
        for link in partnet_link_obj.get_links():
            # Get Unique ID from ManiSkill simulation
            link_id = link._objs[0].entity.per_scene_id

            # Fill Class Map
            self.segm_id_to_class_id_map[link_id] = self.anno_handler.get_class_id(
                class_name
            )

            affords = linkname_to_affordances.get(link.name, [])
            self._register_affordances(link_id, affords)

    def _register_affordances(self, obj_id, affordance_list: list):
        aff_bitlist = self.anno_handler.zero_aff()
        for affordance in affordance_list:
            aff_bitlist[self.affordance_id_map[affordance]] = 1

        self.segm_id_to_affordance_id_map[obj_id] = np.asarray(aff_bitlist, dtype=bool)

    def register_missing_objects(self, env: BaseEnv):
        # Add Missing Objects
        for obj_id, obj in env.segmentation_id_map.items():
            # Skip Already Registered Objects
            if obj_id in self.segm_id_to_class_id_map:
                continue

            parsed_name, sim_name = parse_name(obj)
            class_name = self.anno_handler.check_if_known(parsed_name, sim_name)

            self.segm_id_to_class_id_map[obj_id] = self.anno_handler.get_class_id(
                class_name
            )
            self._register_affordances(
                obj_id, self.anno_handler.get_affordance_list(class_name)[0]
            )

    def get_affordance_list(self, obj_id):
        if obj_id in self.segm_id_to_affordance_id_map:
            return self.segm_id_to_affordance_id_map[obj_id]

        if obj_id in self.segm_id_to_class_id_map:
            class_name = self.segm_id_to_class_id_map[obj_id]
            return self.anno_handler.get_affordance_list(class_name)[1]

        return self.anno_handler.zero_aff()
