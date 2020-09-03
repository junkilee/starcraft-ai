import os

DEFAULT_BACKEND = {
    'ppo': 'pytorch'
}

DEFAULT_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'data')

FORCE_DATESTAMP = False

DEFAULT_SHORTHAND = True

WAIT_BEFORE_LAUNCH = 5

