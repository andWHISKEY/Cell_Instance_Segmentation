
"""Getting params from the command line."""

import argparse

def parameter_parser():
   
    parser = argparse.ArgumentParser(description="Run CellSegmentation.")

    parser.add_argument("--epochs",
                        type=int,
                        default=14,
	                help="Number of training epochs. Default is 14.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=48,
	                help="Number of data per batch. Default is 48.")

    parser.add_argument("--model",
                        type=str,
                        default="resnet34",
	                help="Model name.")


    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--num-workers",
                    type=int,
                    default=0,
                help="Num workers. Default is 0.")


    parser.set_defaults(pin_memory=True)

    parser.add_argument("--drop-last",
                    dest="drop_last",
                    action="store_true")

    parser.set_defaults(drop_last=False)

    return parser.parse_args()
