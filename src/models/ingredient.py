from src.models import __dict__

def get_model(args):
    return __dict__[args.arch](num_classes_train=args.num_classes_train)