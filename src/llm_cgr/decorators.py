"""Useful decorators for research methods."""


def experiment(function):
    """Decorator to mark a function as an experiment."""

    def wrapper(*args, **kwargs):
        print(f"===== STARTING EXPERIMENT === {function.__name__} =====")
        result = function(*args, **kwargs)
        print(f"===== FINISHED EXPERIMENT === {function.__name__} =====")
        return result

    return wrapper
