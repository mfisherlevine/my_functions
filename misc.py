import types


def imports():
    for name, val in globals().items():
        # module imports
        if isinstance(val, types.ModuleType):
            yield name, val
        # functions / callables
        if hasattr(val, '__call__'):
            yield name, val


noglobal = lambda fn: types.FunctionType(fn.__code__, dict(imports()))

# use as:
# a = 1
# @noglobal
# def f(b):
#     print(a)


def retrieve_name(var):
    import inspect

    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
