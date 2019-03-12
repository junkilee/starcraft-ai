Installation
============

.. role:: bash(code)
   :language: bash

Install Starcraft 2
-------------------
 - Mac OS X or Windows
   - Follow the below link and find 'Play Free Now' button and press to install the game.
   - https://starcraft2.com/en-us/
 - Linux
   - Follow the below link and follow the instructions to down the Linux version of the Starcraft.
   - https://github.com/Blizzard/s2client-proto#downloads
 - Brown Department Machine
   - Use the below path to find the Starcraft II.
     - :bash:`/research/xai/starcraft/StarCraftII`
   - Add the above path to your environment variable 'SC2PATH'.
     - :bash:`export SC2PATH='/research/xai/starcraft/StarCraftII'`

Install miniconda
-----------------
 - Go to the below site and follow the instructions to install Miniconda 3 on your platform.
   - https://conda.io/miniconda.html
 - Create your starcraft environment
   - :bash:`conda create -n starcraft python=3`
 - Install other necessary components if necessary

Install pysc2
-------------
 - Please install it from the source code
   - http://github.com/deepmind/pysc2
   - :bash:`pip install -e .`
