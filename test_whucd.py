import argparse
import glob
import os

import lightning
import pandas as pd
from src.change_detection import ChangeDetectionTask
from src.datasets.whucd import WHUCDDataModule
from tqdm import tqdm


def main(args):
    lightning.seed_everything(0)

    checkpoints = glob.glob(f"{args.ckpt_root}/**/checkpoints/epoch*.ckpt", recursive=True)
    runs = [ckpt.split(os.sep)[-3] for ckpt in checkpoints]

    metrics = {}
    for run, ckpt in tqdm(zip(runs, checkpoints, strict=False), total=len(runs)):
        # Initialize the DataModule
        datamodule = WHUCDDataModule(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        # Load model
        module = ChangeDetectionTask.load_from_checkpoint(ckpt, map_location="cpu")

        # Create trainer (use CPU fallback if no GPU)
        trainer = lightning.Trainer(
            accelerator=args.accelerator,
            devices=1 if args.accelerator == "cpu" else [args.device],
            logger=False,
            precision="16-mixed" if args.accelerator != "cpu" else 32
        )

        # Run test
        test_result = trainer.test(model=module, datamodule=datamodule)
        metrics[run] = test_result[0]
        metrics[run]["model"] = module.hparams.model

    # Save results
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv(args.output_filename)
    print(f"✅ Metrics saved to {args.output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/content/whucd_prepared")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="cpu")  # 'gpu' if available
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-filename", type=str, default="pretrained_results.csv")
    args = parser.parse_args()
    main(args)
