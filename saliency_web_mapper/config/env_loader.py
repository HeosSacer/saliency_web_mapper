from os import environ as env
from saliency_web_mapper.config.environment import SaliencyWebMapperEnvironment
import logging


def env_loader(disable_warnings: bool = False) -> SaliencyWebMapperEnvironment:
    """
    loads environment variables into a class:
    E.g: USER -> class.user
    """
    args = SaliencyWebMapperEnvironment()
    for key in env.keys():
        key = key.lower()
        if key in dir(args):
            data_type: type = type(getattr(args, key))
            if data_type == bool:
                if env[key.upper()].lower() == "false":
                    args.__setattr__(key, False)
                else:
                    args.__setattr__(key, True)
            else:
                # Watch out that str can be converted to target type
                args.__setattr__(key, data_type(env[key.upper()]))
    if not disable_warnings:
        for key in dir(args):
            if not key.upper() in env.keys() and not key.startswith('_'):
                logging.warning(f"environment variable {key} not set, using default setting value {args.__getattribute__(key)} for {key}.")
    return args


if __name__ == "__main__":
    args = env_loader()
