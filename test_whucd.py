import argparse
import glob
import os

import lightning as L
import pandas as pd
import torch
from tqdm import tqdm
from src.change_detection import ChangeDetectionTask
from src.datasets.whucd import WHUCDDataModule


def main(args):
    L.seed_everything(0)

    # Support individual ckpt file
    if os.path.isfile(args.ckpt_root):
        checkpoints = [args.ckpt_root]
        runs = ["run1"]
    else:
        checkpoints = glob.glob(f"{args.ckpt_root}/**/checkpoints/*.ckpt", recursive=True)
        runs = [os.path.basename(os.path.dirname(os.path.dirname(ckpt))) for ckpt in checkpoints]

    metrics = {}
    for run, ckpt in tqdm(zip(runs, checkpoints), total=len(checkpoints)):
        datamodule = WHUCDDataModule(
            root=args.root, batch_size=args.batch_size, patch_size=256, num_workers=args.workers
        )
        model = ChangeDetectionTask(model="unet", backbone="resnet50")

        # Load manually if it's a weights-only .ckpt
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

        trainer = L.Trainer(
            accelerator=args.accelerator, devices=[args.device], logger=False, precision="16-mixed"
        )
        metrics[run] = trainer.test(model=model, datamodule=datamodule)[0]
        metrics[run]["model"] = model.hparams.model

    metrics = pd.DataFrame.from_dict(metrics, orient="index")
    metrics.to_csv(args.output_filename)
    print(f"✅ Results saved to {args.output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/content/whucd_prepared")
    parser.add_argument("--ckpt-root", type=str, default="/content/unet_resnet50(1).ckpt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-filename", type=str, default="metrics.csv")
    args = parser.parse_args()
    main(args)
