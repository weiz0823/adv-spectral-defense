import os
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
from tools.common import *


def main(args: CLIConfig = None):
    if args is None:
        args = configs.get_args()
    if args.attack == "patch":
        from tools.train_patch import main as train_patch_main

        train_patch_main(args)
    elif args.attack == "render":
        from tools.train_camou import main as train_camou_main

        train_camou_main(args)
    elif args.attack == "noise":
        from tools.train_noise import main as train_noise_main

        train_noise_main(args)
    else:
        raise ValueError(f"Unknown attack type: {args.attack}")


if __name__ == "__main__":
    main()
