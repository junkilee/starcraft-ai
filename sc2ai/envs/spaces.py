from gym import Space


class ListActionSpace(Space):
    """
    A list of simpler action spaces

    Args:
        spaces: A list of tuples. Each tuple contains name and sub action space.
    """
    def __init__(self, spaces):
        super().__init__()
        if isinstance(spaces, dict):
            spaces = list(spaces.items())
        elif isinstance(spaces, list):
            spaces = spaces
        else:
            raise NotImplementedError
        for item in spaces:
            if not isinstance(item, tuple):
                raise Exception("The list items should be tuple.")
            if not isinstance(item[0], str):
                raise Exception("The first item in each tuple should be a string.")
            if not isinstance(item[1], (Space, None.__class__)):
                raise Exception("The second item in each tuple should be a Space object or None.")

        self.spaces = spaces
        self.n = len(spaces)
        print(self.n)

    def sample(self):
        # first sample which action to pick
        action = self.np_random.randint(self.n)
        # sample the sub action space (parameters for the main action)
        if self.spaces[action][1] is None:
            parameters = None
        else:
            parameters = self.spaces[action][1].sample()
        return (action, parameters)

    def contains(self, x):
        if not isinstance(x, tuple):
            return False
        if len(x) is not 2:
            return False
        if not isinstance(x[0], int):
            return False
        if x[0] >= 0 and x[0] < self.n:
            if self.spaces[x] is None or self.spaces[x].contains(x[1]):
                return True
        return False

    def _get_shape(self):
        """Concatenate all the parameters
        """
        shape = (self.n,)
        for i in range(self.n):
            if self.spaces[i] is not None:
                shape += self.spaces[i].shape()
        return shape

    def __repr__(self):
        output = "ListActionSpace("
        for i in range(self.n):
            output += self.spaces[i][0] + "(" + str(self.spaces[i][1]) + ")"
        output += ")"
        return output

    def to_jsonable(self, sample_n):
        json_list = []
        for sample in sample_n:
            action = sample[0]
            parameters = sample[1]
            if action < 0 or action >= self.n:
                raise Exception("The action id in the samples is out of bound.")
            if parameters is None:
                parameters = []
            else:
                parameters = self.spaces[action][1].to_jsonable([parameters])
            json_list += [{'action':action, 'parameters':parameters}]
        return json_list

    def from_jsonable(self, sample_n):
        samples = []
        for sample in sample_n:
            action = sample['action']
            if action < 0 or action >= self.n:
                raise Exception("The action id in the samples is out of bound.")
            parameters = sample['parameters']
            if parameters is []:
                parameters = None
            else:
                parameters = self.spaces[action][1].from_jsonable([parameters])

            samples += [(action, parameters)]
        return samples
