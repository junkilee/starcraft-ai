import numpy as np
from gltl_tools.gltl.gltl import gltl_to_mdp
from gltl_tools.gltl.utils import mdp_to_dict_matrix, return_ordered_aps

class GLTLMDP:
    """ A class representing the mental state of an agent using GLTL. It manages transitions between state based on a list of propositions.
    """

    def __init__(self, expression, propositionDict):
        # { 'atBeacon': atBeacon
        # }

        self.mdp = gltl_to_mdp(expression)
        self.propositionDict = propositionDict
        self.currentState = self.mdp.init
        self.transitionMatrices = mdp_to_dict_matrix(self.mdp)
        self.aps = return_ordered_aps(self.mdp)

        # make sure that APs are a subset of proposition dict
        assert(set(self.aps) in propositionDict)

    def transition(self, timestep):
        results_dict = classify(timestep)
        transitionMatrix = get_next_trans_mat(self.transition_mats_dict, results_dict, self.ordered_aps)
        transitionRow = transitionMatrix[self.currentState]
        self.currentState = np.random.choice(len(transitionRow), 1, p=transitionRow)

def get_ap_combo_key(classifier_dict, ordered_aps):
    """
    Method to get appropriate transition matrix for classification of obs.
    :param classifier_dict: dictionary of {proposition: True/False}
    :param ordered_aps: list of atomic propositions in order of the {1, 0} n-tuple that
    represents the symbol of the env state.
    :return: tuple of length len(ordered_aps) where the ith value represents the ith AP
    is true or false.
    """
    if all(ap in classifier_dict for ap in ordered_aps):
        return tuple([classifier_dict[ap] for ap in ordered_aps])
    else:
        raise ValueError("APs must be either `belowground`, `logleft`, "
                         "`logtouching`, or `walltouching`.")


def get_next_trans_mat(transition_mats_dict, results_dict, ordered_aps):
    """
    Method to return the appropriate right transition matrix of
    dimensions [num_task_states, num_task_states], given the results of the classifier and
    list of ordered APs.
    :param transition_mats_dict: dict from symbol to transition matrix (symbol is a n-tuple of 1s and 0s)
    :param results_dict: dict from {AP: bool}
    :param ordered_aps: list of AP in order of how they should appear in tuple
    :return: a right transition matrix of dimensions [num_task_states, num_task_states].
    """
    return transition_mats_dict[get_ap_combo_key(results_dict, ordered_aps)]
