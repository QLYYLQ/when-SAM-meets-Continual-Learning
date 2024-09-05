prompt_encoder_module_entrypoints = {}


def register_module(fn):
    model_name_split = fn.__module__.split('.')
    model_name_split.append(fn.__name__)
    dataset_name = ".".join(model_name_split[-2:])
    prompt_encoder_module_entrypoints[dataset_name] = fn
    return fn
