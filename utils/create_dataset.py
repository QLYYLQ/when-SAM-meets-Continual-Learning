from .init_from_config import auto_init
def load_dataset_from_config(config):
    pass

def create_task_dataset(config,task_number):
    task = config.training[task_number]
    design = task.design
    index_order = task.index_order
    dataset = {}
    labels_old = []
    for key,value in index_order.items():
        labels = value


