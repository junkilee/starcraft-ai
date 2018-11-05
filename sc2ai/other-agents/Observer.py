
import enum

class PlayerRelative(enum.IntEnum):
  """The values for the `player_relative` feature layers."""
  NONE = 0
  SELF = 1
  ALLY = 2
  NEUTRAL = 3
  ENEMY = 4

class FeatureUnit(enum.IntEnum):
  """Indices for the `feature_unit` observations."""
  unit_type = 0
  alliance = 1
  health = 2
  shield = 3
  energy = 4
  cargo_space_taken = 5
  build_progress = 6
  health_ratio = 7
  shield_ratio = 8
  energy_ratio = 9
  display_type = 10
  owner = 11
  x = 12
  y = 13
  facing = 14
  radius = 15
  cloak = 16
  is_selected = 17
  is_blip = 18
  is_powered = 19
  mineral_contents = 20
  vespene_contents = 21
  cargo_space_max = 22
  assigned_harvesters = 23
  ideal_harvesters = 24
  weapon_cooldown = 25
  order_length = 26  # If zero, the unit is idle.
  tag = 27  # Unique identifier for a unit (only populated for raw units).


class Observer:

    def __init__(self):
        #Stores tags for all allied units. Updated as units or are created in game
        self.allied_units = []
        pass

    def print_obs(self,obs):
        """prints the whole observation from this timestep of the sc2 game"""
        print(obs)

    def get_total_health(self,obs):
        """returns the total health of all allied units"""
        total_health = 0
        for unit in obs.observation.raw_units:
            if(unit.alliance == PlayerRelative.SELF):
                total_health += unit[FeatureUnit.health]
        return total_health


    def get_center_of_mass_allies(self,obs):
        """returns the x,y of the center of mass for allied units
        returns [-1,-1] if no allied units are present"""
        num_units = 0
        total_x = 0
        total_y = 0
        for unit in obs.observation.raw_units:
            if(unit.alliance == PlayerRelative.SELF):
                total_x += unit[FeatureUnit.x]
                total_y += unit[FeatureUnit.y]
                num_units += 1
        if num_units != 0:
            return [total_x/num_units,total_y/num_units]
        else:
            return [-1,-1]

    def get_center_of_mass_enemies(self,obs):
        """returns the x,y of the center of mass for the enemy units
        returns [-1,-1] if no enemies units are present
        """
        num_units = 0
        total_x = 0
        total_y = 0
        for unit in obs.observation.raw_units:
            if(unit.alliance == PlayerRelative.ENEMY):
                total_x += unit[FeatureUnit.x]
                total_y += unit[FeatureUnit.y]
                num_units += 1
        if num_units != 0:
            return [total_x/num_units,total_y/num_units]
        else:
            return [-1,-1]

    def get_total_shield(self,obs):
        """returns total shield of allied units"""

    def list_all_tags(self,obs):
        """return a list of all tags for all units in the current game"""

    def get_unit(self,tag):
        """Takes in a units tag number and returns the unit corresponding
        that tag"""
