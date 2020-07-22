from saliency_web_mapper.__main__ import app
from saliency_web_mapper.config.env_loader import env_loader


if __name__ == "__main__":
    args = env_loader()

    if args.debug:
        import pydevd_pycharm
        from time import sleep
        # Wait for debug server
        while True:
            try:
                pydevd_pycharm.settrace(args.debug_address, port=2376, stdoutToServer=True, stderrToServer=True)
                break
            except:
                sleep(1)

    app(args)