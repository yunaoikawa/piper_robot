from multiprocessing import Process
import signal
import importlib

import hydra
import omegaconf

processes: list[Process] = []


def sigint_handler(signum, frame):
    for p in processes:
        p.terminate()
    print("Robot server shutdown gracefully.")
    exit(0)


signal.signal(signal.SIGINT, sigint_handler)


@hydra.main(version_base="1.2", config_path="launch", config_name="run")
def main(cfg: omegaconf.DictConfig):
    for node in cfg["nodes"]:
        try:
            fn = importlib.import_module(node["module"]).main
        except ImportError:
            print(f"Module {node['module']} not found.")
            exit(1)

        if "args" in node:
            p = Process(target=fn, kwargs=node["args"], name=node["name"])
        else:
            p = Process(target=fn, name=node["name"])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
