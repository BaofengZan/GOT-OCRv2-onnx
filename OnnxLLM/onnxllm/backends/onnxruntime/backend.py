import onnxruntime as rt


class OnnxRunner:
    def __init__(self, model_path):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        self.model = rt.InferenceSession(model_path)
        self.input_names = [input.name for input in self.model.get_inputs()]
        self.output_names = [output.name for output in self.model.get_outputs()]

    def __call__(self, *args):
        kwargs = {param: arg for param, arg in zip(self.input_names, args)}
        output = self.model.run(self.output_names, kwargs)
        if len(self.output_names) == 1:
            return output[0]

        return output
