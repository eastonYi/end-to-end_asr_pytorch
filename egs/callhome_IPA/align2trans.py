"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from tqdm import tqdm


def load_vocab(path):
    idx2token = {}
    with open(path) as f:
        for line in f:
            token, idx = line.strip().split()
            if '#' in token:
                break
            token = token.split('_')[0]
            idx2token[idx] = token

    return idx2token


def align_shrink(align):
    _token = None
    list_tokens = []
    for token in align:
        if _token != token:
            list_tokens.append(token)
            _token = token

    return list_tokens


def main(ali, phone2idx, output):
    idx2token = load_vocab(phone2idx)

    with open(ali) as f, open(output, 'w') as fw:
        for line in tqdm(f):
            uttid, align = line.strip().split(maxsplit=1)
            phones = align_shrink(align.split())
            line = uttid + ' ' + ' '.join(idx2token[p] for p in phones)
            fw.write(line + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ali', type=str, dest='ali')
    parser.add_argument('--phones', type=str, dest='phones')
    parser.add_argument('--output', type=str, dest='output')
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    main(args.ali, args.phones, args.output)
    logging.info("Done")
