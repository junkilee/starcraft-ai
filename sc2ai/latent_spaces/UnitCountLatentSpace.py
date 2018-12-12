
#Import enums for units, buildings, maybe resources?
import pysc2.lib.units as units
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






class UnitCountLatentSpace():

    def __init__(self):
        """Creates intial dictionary for unit counts"""
        self.unitCounts = {}
        #Initializes unitCounts dictionary
        for unit in units.Neutral:
            self.unitCounts[unit] = 0
        for unit in units.Protoss:
            self.unitCounts[unit] = 0
        for unit in units.Terran:
            self.unitCounts[unit] = 0
        for unit in units.Zerg:
            self.unitCounts[unit] = 0

    def update(self,obs):
        """Zeroes out dictionary and then refills
        it based on observation from current timestep"""
        #Zero out dictionary
        for unit in units.Neutral:
            self.unitCounts[unit] = 0
        for unit in units.Protoss:
            self.unitCounts[unit] = 0
        for unit in units.Terran:
            self.unitCounts[unit] = 0
        for unit in units.Zerg:
            self.unitCounts[unit] = 0
        #Loops through and creates array of current unit counts
        for unit in obs.observation.feature_units:
            if(unit.alliance == PlayerRelative.SELF):
                self.unitCounts[unit["unit_type"]] += 1
