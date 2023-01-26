# merfish-blank-barcode-generator

## Description

Script to generate blank barcodes for a MERFISH experiment given an existing codebook.

## Getting Started

### Dependencies
* NumPy
* Pandas
* SciPy (for computing Hamming distance - Could be remade using NumPy)
* tqdm
* matplotlib (For plotting functionality)
* Seaborn (For plotting functionality)

### Executing program
* How to run the program
* Step-by-step bullets

```
merfishBarcodeBlankGenerator.py [-h] -i INPUT_FILE -d MIN_DIST [-b NUM_BLANKS] [-n LENGTH] [-w WEIGHT]
                                       [-o OUTPUT_FILE] [-p] [-v]
options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input .csv file
  -d MIN_DIST, --min_dist MIN_DIST
                        Minimum Hamming distance between barcodes in codebook.
  -b NUM_BLANKS, --num_blanks NUM_BLANKS
                        The number of blank barcodes to generate. Default is 5.
  -n LENGTH, --length LENGTH
                        Binary code of length N. If not provided, this will be inferred from codebook.
  -w WEIGHT, --weight WEIGHT
                        Hamming weight of desired barcodes. If not provided, this will be inferred from codebook.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file. The old codebook with the blank barcodes appended.
  -p, --plot
  -v, --verbose
```

## Authors

Jakob Rask Johansen

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
