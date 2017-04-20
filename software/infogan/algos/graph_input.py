class GraphOutputsManager(object):
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.d = {}  # {function: GraphOutput object}

    def get_output_tensor(self, function, key=None):
        default_graph_output = GraphOutput(self.input_tensor, function, key)
        graph_output = self.d.setdefault(function.__name__, default_graph_output)
        return graph_output.get_output_tensor()


class GraphOutput(object):
    def __init__(self, input_tensor, function, key=None):
        self.input_tensor = input_tensor
        self.function = function
        self.key = key
        self.output_tensor = None

    def get_output_tensor(self):
        if self.output_tensor is None:
            msg = 'Applying function "{}" to tensor "{}"'
            print(msg.format(self.function.__name__, self.input_tensor.name))
            if self.key is not None:
                kwargs = {self.key: self.input_tensor}
                self.output_tensor = self.function(**kwargs)
            else:
                self.output_tensor = self.function(self.input_tensor)
        return self.output_tensor
