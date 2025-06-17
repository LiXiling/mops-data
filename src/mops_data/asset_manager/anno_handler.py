from enum import IntEnum
from importlib import resources

import numpy as np
import pandas as pd

SINGLETON_ANNOATION_HANDLER = None


def load_annotations():
    global SINGLETON_ANNOATION_HANDLER
    if SINGLETON_ANNOATION_HANDLER is None:
        SINGLETON_ANNOATION_HANDLER = AnnotationHandler()
    return SINGLETON_ANNOATION_HANDLER


class AnnotationHandler:
    def __init__(self):

        with resources.files("mops_data.resources").joinpath(
            "class_affordances.json"
        ).open() as f:
            self.class_affordance_df = pd.read_json(f, orient="records")

        with resources.files("mops_data.resources").joinpath(
            "partnet-mobility_affordances.json"
        ).open() as f:
            self.partnet_mobility_df = pd.read_json(f, orient="records")

        classes = ["Background"] + self.class_affordance_df[
            "model_cat"
        ].unique().tolist()
        self.all_class_names = classes
        self.class_id_map = {classes[i]: i for i in range(len(classes))}
        self.affordance_id_map = None
        self._fill_affordance_info()

    def _fill_affordance_info(self):
        # Merge all list entries into a set
        all_affordances = set(
            affordance_entry
            for row in self.class_affordance_df["affordances"]
            for affordance_entry in row
        )

        self.affordance_id_map = IntEnum(
            "Affordance", {val: idx for idx, val in enumerate(sorted(all_affordances))}
        )

    def get_annotated_classes(self):
        return self.class_affordance_df["model_cat"].unique()

    def _add_class(self, class_name):
        if class_name not in self.all_class_names:
            self.all_class_names.append(class_name)
            self.class_id_map[class_name] = len(self.all_class_names) - 1
        return len(self.all_class_names) - 1

    def get_class_id(self, class_name):
        if class_name in self.class_id_map:
            return self.class_id_map[class_name]
        return self._add_class(class_name)

    def get_affordance_list(self, class_name):
        affords = []
        if class_name in self.get_annotated_classes():
            affordances = self.class_affordance_df[
                self.class_affordance_df["model_cat"] == class_name
            ]["affordances"]
            affords = affordances.values[0]

        aff_id_list = self.zero_aff()
        for aff in affords:
            aff_id_list[self.affordance_id_map[aff]] = 1

        return affords, np.asarray(aff_id_list)

    def get_aff_name(self, id: int):
        return self.affordance_id_map(id).name

    def get_affordance_id(self, affordance_name):
        return self.affordance_id_map[affordance_name]

    def check_if_known(self, parsed_name, sim_name):
        if sim_name in self.all_class_names:
            return sim_name
        return parsed_name

    def zero_aff(self):
        return np.zeros(len(self.affordance_id_map))


if __name__ == "__main__":
    obj = AnnotationHandler()
    print(obj.class_affordance_df)
