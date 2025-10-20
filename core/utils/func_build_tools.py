import inspect
from typing import get_type_hints

def build_tools_from_functions(funcs):
    tool_descriptions = []
    tool_look_up = {}
    for func in funcs:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        args = {
            name: hints.get(name, str).__name__
            for name in sig.parameters.keys()
        }

        tool_look_up[func.__name__] = func
        tool_descriptions.append({
            "name": func.__name__,
            "description": (func.__doc__ or "No description.").strip(),
            "arguments": args
        })

    return tool_descriptions, tool_look_up

def get_args_in_order(func, args_dict):
    return [args_dict[name] for name in inspect.signature(func).parameters if name in args_dict]
