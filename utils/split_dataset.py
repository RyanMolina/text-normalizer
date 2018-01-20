import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="filename of dataset")
    parser.add_argument("--out", type=str, default="dataset",
                        help="output location")
    parser.add_argument("--test_size", type=float, default=None,
                        help="size of test dataset (0 - 1.0)")
    parser.add_argument("--dev_size", type=float, default=None,
                        help="size of dev dataset (0 - 1.0)")
    parser.add_argument("--train_size", type=int, default=None,
                        help="size of train dataset")
    parser.add_argument("--ext", type=str, required=True,
                        help="output file extension")

    args = parser.parse_args()


def main():
    split(args.src, args.out, args.ext, test_size=args.test_size,
          train_size=args.train_size, dev_size=args.dev_size)


def split(src, tgt, file_ext, test_size=None, train_size=None, dev_size=None):
    with open(src, 'r') as datafile:
        dataset = datafile.read()
        dataset = dataset.split('\n')

    if not train_size:
        train_size = len(dataset)

    if not test_size:
        test_size = int(len(dataset) * 0.10)

    if not dev_size:
        dev_size = int(len(dataset) * 0.10)

    test_set = dataset[:test_size]
    dev_set = dataset[test_size:dev_size + test_size]
    train_set = dataset[dev_size + test_size:train_size]

    split_data(tgt + '/test.' + file_ext, test_set)
    split_data(tgt + '/train.' + file_ext, train_set)
    split_data(tgt + '/dev.' + file_ext, dev_set)


def split_data(filename, set):
    with open(filename, 'w') as f:
        for row in set:
            print(row, file=f)
