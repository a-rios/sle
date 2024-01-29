import argparse
import pandas as pd
from sle.utils import smooth_labels



def main(in_csv, out_csv, num_labels):
    df = pd.read_csv(in_csv) # data with a quantized label column
    df_soft = smooth_labels(df, label_col="y", num_labels=num_labels) # assuming 5 original label values

    df_soft.to_csv(out_csv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", type=str, help="input csv with original labels", metavar="PATH", required=True)
    parser.add_argument("--out-csv", type=str, help="output csv with smoothed labels", metavar="PATH", required=True)
    parser.add_argument("--num-labels", type=int, default=5, help="number of labels", metavar="PATH")


    args = parser.parse_args()
    main(args.in_csv, args.out_csv, args.num_labels)
