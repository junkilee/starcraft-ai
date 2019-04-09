

class GLTLMDP:
    """ A class representing the mental state of an agent using GLTL. It manages transitions between state based on a list of propositions.
    """

    def __init__(self, expression, propositionDict):
        { 'atBeacon': atBeacon 
        }

        self.mdp = libraryCall(expression)
        self.propositionDict = propositionDict
        self.currentState = self.mdp.init
        self.transitionMatrices = libraryCall(self.mdp)
        #TODO make sure that APs are a subset of proposition dict

    def transition(self, timestep):
        truthTuple = [self.propositionDict[key](timestep) for key self.mdp.APs]
        transitionMatrix = self.transitionMatrices[truthTuple]
        transitionRow = transitionMatrix[self.currentState]
        self.currentState = np.random.choice(len(transitionRow), 1, transitionRow)