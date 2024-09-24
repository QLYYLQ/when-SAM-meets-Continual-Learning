
_model_entrypoints = {}

def register_model(fn):
    model_name_split = fn.__module__.split('.')
    model_name_split.append(fn.__name__)
    dataset_name = ".".join(model_name_split[-2:])
    _model_entrypoints[dataset_name] = fn
    return fn


def model_entrypoint(dataset_name):
    return _model_entrypoints[dataset_name]

