import functools
import torch


def input_guard(func):
    """
    Decorator that ensures all tensor inputs are contiguous and on the same device.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert all tensor arguments to contiguous
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.contiguous())
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = value.contiguous()
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper
