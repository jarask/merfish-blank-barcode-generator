"""
Script to generate blank barcodes given an existing codebook
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=__doc__)
parser.add_argument("-i", "--input_file", required=True,
                    help="Input .csv file")
parser.add_argument("-d", "--min_dist", type=int, required=True,
                    help="Minimum Hamming distance between barcodes in codebook.")
parser.add_argument("-b", "--num_blanks", type=int, default=5,
                    help="The number of blank barcodes to generate. Default is 5.")
parser.add_argument("-n", "--length", type=int, required=False, default=None,
                    help="Binary code of length N. If not provided, this will be inferred from codebook.")
parser.add_argument("-w", "--weight", type=int, required=False, default=None,
                    help="Hamming weight of desired barcodes. If not provided, this will be inferred from codebook.")
parser.add_argument("-o", "--output_file", default=None,
                    help="Output file. The old codebook with the blank barcodes appended.")
parser.add_argument("-p", "--plot", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")


def main(args):
    csv_path = Path(args.input_file)
    if not csv_path.exists:
        print(f"Could not find file: {csv_path}")
        raise SystemExit(1)

    # Read in codebook
    codebook = pd.read_csv(csv_path, header=None, dtype=object)
    csv_sep = ","
    if codebook.shape[1] == 1:
        if ";" in codebook[0][0]:
            csv_sep = ";"
            codebook = pd.read_csv(
                csv_path, header=None, sep=";", dtype=object
            )
        elif "," in codebook[0][0]:
            csv_sep = ","
            codebook = pd.read_csv(
                csv_path, header=None, sep=",", dtype=object
            )
    elif codebook.shape[1] == 2:
        pass

    barcodes_codebook = codebook[1].to_list()

    if args.length is not None:
        N = args.length
    else:
        # Get the length of the first barcode
        N = len(barcodes_codebook[0])
    if args.weight is not None:
        W = args.weight
    else:
        # Get the Hamming weight of the first barcode
        W = barcodes_codebook[0].count("1")

    D = args.min_dist  # with minimum distance D
    M = 2**N  # number of unique codes in general

    if args.verbose:
        print(f"N={N} | W={W} | D={D}")

    codes = []
    for i in range(M):
        code = format(i, f"0{N}b")
        if code.count("1") == W:
            codes.append(code)

    # Compute Hamming distances
    hd_df = pd.DataFrame([])
    for i, barcode1 in tqdm(enumerate(codes), total=len(codes), disable=not args.verbose,
                            desc="Computing Hamming distances for candidates"):
        for j, barcode2 in enumerate(barcodes_codebook):
            hd_df.loc[i, j] = hamming(
                np.array(list(barcode1)), np.array(list(barcode2))) * len(barcode1)
    hd_df.columns = [str(barcode) for barcode in barcodes_codebook]
    hd_df.index = [str(barcode) for barcode in codes]

    valid_blanks = hd_df[(hd_df >= D).all(axis=1)]

    # Chech Hamming distances between the blanks and only keep valid ones from these
    # Get only a portion of the data (this is faster and can be good enough)
    resample_blanks = True
    tries = 0
    while resample_blanks:
        if args.verbose:
            print(
                f"\rFinding {args.num_blanks} candidates - Try {tries}", end="")
        if tries == 100:
            print()
            print(
                f"No {args.num_blanks} valid barcodes found in 100 tries. Trying with {args.num_blanks-1} instead.")
            tries = 0
            args.num_blanks -= 1
        if len(valid_blanks) == 0:
            print("No valid blank barcodes could be generated for the given codebook.")
            raise SystemExit(1)
        elif len(valid_blanks) < args.num_blanks:
            # Select all valid blanks, as there are less than desired
            sample_blanks = valid_blanks
            # Do not resample, as we take what we can get
            resample_blanks = False
        else:
            sample_blanks = valid_blanks.sample(n=args.num_blanks)
        hd_blanks = pd.DataFrame([])
        for i, barcode1 in enumerate(sample_blanks.index.to_list()):
            for j, barcode2 in enumerate(sample_blanks.index.to_list()):
                if barcode1 == barcode2:
                    hd_blanks.loc[i, j] = N + 1
                else:
                    hd_blanks.loc[i, j] = hamming(
                        np.array(list(barcode1)), np.array(list(barcode2))) * len(barcode1)
        hd_blanks.index = hd_blanks.columns = [
            str(barcode) for barcode in sample_blanks.index.to_list()]

        # Get the valid blank barcodes
        valid_blanks2 = hd_blanks[(hd_blanks >= D).all(axis=1)]
        # Check if we have enough valid blanks
        if len(valid_blanks2) == args.num_blanks:
            resample_blanks = False
        elif len(valid_blanks2) <= args.num_blanks and not resample_blanks:
            pass
        tries += 1
    print()

    # Visualize the Hamming distance between barcodes
    barcodes_combined = barcodes_codebook + valid_blanks2.index.to_list()
    # Compute Hamming distances
    hd_df2 = pd.DataFrame([])
    for i, barcode1 in enumerate(barcodes_combined):
        for j, barcode2 in enumerate(barcodes_combined):
            hd_df2.loc[i, j] = hamming(
                np.array(list(barcode1)), np.array(list(barcode2))) * len(barcode1)
    hd_df2.index = hd_df2.columns = [
        str(barcode) for barcode in barcodes_combined]

    # Add the blank barcodes to the codebook
    for i, barcode in enumerate(valid_blanks2.index.to_list()):
        codebook.loc[len(codebook)] = [f"BLANK0{i+1}", barcode]
    # Save the new codebook
    if args.output_file is not None:
        codebook.to_csv(args.output_file, sep=";", index=False, header=False)
    else:
        args.output_file = f"{csv_path.parent}{csv_path.root}{csv_path.stem}_withblanks.csv"
        codebook.to_csv(args.output_file,
                        sep=csv_sep, index=False, header=False)

    if args.verbose:
        print(f"Valid blank barcodes: {valid_blanks2.index.to_list()}")
        print(f"Found in {tries} tries.")
        print(f"New codebook saved at {args.output_file}.")

    if args.plot:
        plot(valid_blanks=valid_blanks2,
             valid_blanks2=hd_blanks,
             hd_df2=hd_df2,
             N=N)


def plot(valid_blanks, valid_blanks2, hd_df2, N):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Clean up the two valid_blanks dataframes
    valid_blanks[valid_blanks == N + 1] = 0
    valid_blanks2[valid_blanks2 == N + 1] = 0

    # Plotting
    sns.set()  # Set seaborn as default style
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Plot the Hamming distances in a heatmap
    sns.heatmap(valid_blanks, cmap="inferno", vmin=0, annot=True,
                square=True, ax=axes[0]).set(title="Valid blanks")
    sns.heatmap(valid_blanks2, cmap="inferno", vmin=0, vmax=8, annot=True,
                square=True, ax=axes[1]).set(title="Valid blanks - Inter-HD")
    # Plot the Hamming distances in a heatmap
    sns.heatmap(hd_df2, cmap="inferno", annot=True, square=True, ax=axes[2]).set(
        title="Inter-Hamming distance")
    plt.show()


if __name__ == "__main__":
    # For debugging
    # sys.argv = sys.argv + ["-i", "C:/Users/jakobrask/Documents/merfish-analysis/misc-testing/data/Codebook_brewer_8r.csv",
    #                        "-d", "4", "-b", "10", "-w", "4", "-p", "-v"]
    args = parser.parse_args()
    # For debugging
    # if args.verbose:
    #     print(" ".join(sys.argv))
    main(args)
