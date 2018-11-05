

class unitInfo:

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

    def __init__(self):




    def print_obs(obs):
        """prints the whole observation from this timestep of the sc2 game"""
        print(obs)
