"""
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
Modified by Easton.
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from tqdm import tqdm


def load_vocab(path, vocab_size=None):
    with open(path, encoding='utf8') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    id_unk = vocab.index('<unk>')
    token2idx = defaultdict(lambda: id_unk)
    idx2token = defaultdict(lambda: '<unk>')
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})
    if '<space>' in vocab:
        idx2token[token2idx['<space>']] = ' '
    if '<blk>' in vocab:
        idx2token[token2idx['<blk>']] = ''
    # if '<pad>' in vocab:
    #     idx2token[token2idx['<pad>']] = ''
    if '<unk>' in vocab:
        idx2token[token2idx['<unk>']] = '<UNK>'

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    uttid a b c d

    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    with open(fpath, encoding='utf-8') as f:
        for l in f:
            words = l.strip().split(',')[1].split()
            word2cnt.update(Counter(words))
    word2cnt.update({"<pad>": 1000000000,
                     "<unk>": 100000000,
                     "<sos>": 10000000,
                     "<eos>": 1000000,
                     "<blk>": 0,})
    with open(fname, 'w', encoding='utf-8') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(fname, len(word2cnt)))


def pre_processing(fpath, fname):
    import re
    with open(fpath, errors='ignore') as f, open(fname, 'w') as fw:
        for line in tqdm(f):
            line = line.strip().split(maxsplit=1)
            idx = line[0]
            # list_tokens = re.findall('\[[^\[\]]+\]|[a-zA-Z0-9^\[^\]]+|[^x00-xff]', line[1])
            list_tokens = re.findall('\[[^\[\]]+\]|[^x00-xff]|[A-Za-z]', line[1])
            list_tokens = [token.upper() for token in list_tokens]

            fw.write(idx+' '+' '.join(list_tokens)+'\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, dest='src_vocab')
    # parser.add_argument('--dst_vocab', type=str, dest='dst_vocab')
    parser.add_argument('--input', type=str, dest='src_path')
    # parser.add_argument('--dst_path', type=str, dest='dst_path')
    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    make_vocab(args.src_path, args.src_vocab)
    # pre_processing(args.src_path, args.src_vocab)
    logging.info("Done")
