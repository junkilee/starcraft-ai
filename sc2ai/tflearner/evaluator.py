import numpy as np
import math


class Evaluator:
    @classmethod
    def at_beacon(cls, timestep):
        """
        Function to check if MoveToBeacon agent
        has reached the beacon
        Params: timestep from game
        Output: True or False
        """
        marine = None
        beacon = None
        # minimum distance required to return true
        min_dist = 8
        for unit in timestep.observation.feature_units:
            if unit["unit_type"] == 48:
                marine = unit
            elif unit["unit_type"] == 317:
                beacon = unit
        coordinates = np.asarray([(marine["x"],marine["y"]),(beacon["x"],beacon["y"])])
        if np.linalg.norm(coordinates[0]-coordinates[1]) < min_dist:
            return True
        else:
            return False

    def get_results_dict(self, timestep):
        return {"atbeacon": self.at_beacon(timestep)}
