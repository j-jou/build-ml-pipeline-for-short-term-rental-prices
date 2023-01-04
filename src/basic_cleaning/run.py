#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Downloading artifact {args.input_artifact}")
    local_path = run.use_artifact(args.input_artifact).file()

    # Readinf as dataframe
    df = pd.read_csv(local_path)

    # Drop outliers
    logger.info(f"Dropping Outliers between {args.min_price} and {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # save cleaned data as csv
    logger.info(f"Saving cleaned data to {args.output_artifact}")
    df.to_csv(args.output_artifact, index=False)

    # Storing in W&B
    logger.info(f"Logging artifact {args.output_artifact} in W&B") 
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,)
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="csv file from W&B",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='cleaned csv',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help='type of output',
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help='short desription',
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='min price below which data is considered as outlier',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='max price below which data is considered as outlier',
        required=True
    )


    args = parser.parse_args()

    go(args)
