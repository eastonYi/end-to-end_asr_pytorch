import torch
import os


def main(dir_model, num):
    # sum
    files = []
    for r, d, f in os.walk(dir_model):
        for file in f:
            if 'model' in file and 'avg' not in file:
                files.append(os.path.join(r, file))

    files.sort()
    files2avg = files[-num:]
    files2rm = files[:-num]
    [os.remove(i) for i in files2rm]

    avg = None
    for path in files2avg:
        print('load model', path)
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in states.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] /= num

    save_path = os.path.join(dir_model, 'avg.model')
    torch.save(avg, save_path)
    print('save avg model', save_path)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, dest='dir', default=None)
    parser.add_argument('--num', type=int, dest='num', default=None)

    param = parser.parse_args()

    main(param.dir, param.num)
