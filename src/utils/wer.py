#!/usr/bin/env python

# Word(Character) Error Rate, also gives the alignment infomation.
# Author: XiaoRui Wang <xrwang@hitic.ia.ac.cn>

import sys, re, os.path
import argparse

# Cost of alignment types
SUB_COST = 3
DEL_COST = 3
INS_COST = 3

CRT_ALIGN = 0
SUB_ALIGN = 1
DEL_ALIGN = 2
INS_ALIGN = 3
END_ALIGN = 4


align_name = ['crt', 'sub', 'del', 'ins', 'end']

class entry:
    'Alignment chart entry, contains cost and align-type'

    def __init__(self, cost = 0, align = CRT_ALIGN):
        self.cost = cost
        self.align = align

def getidx(name):
    name = name.strip()
    index = os.path.basename(name)
    (index, ext) = os.path.splitext(index)
#    print index,ext
    return index


def distance(ref, hyp):
    ref_len = len(ref)
    hyp_len = len(hyp)

    chart = []
    for i in range(0, ref_len + 1):
        chart.append([])
        for j in range(0, hyp_len + 1):
            chart[-1].append(entry(i * j, CRT_ALIGN))

    # Initialize the top-most row in alignment chart, (all words inserted).
    for i in range(1, hyp_len + 1):
        chart[0][i].cost = chart[0][i - 1].cost + INS_COST;
        chart[0][i].align = INS_ALIGN
    # Initialize the left-most column in alignment chart, (all words deleted).
    for i in range(1, ref_len + 1):
        chart[i][0].cost = chart[i - 1][0].cost + DEL_COST
        chart[i][0].align = DEL_ALIGN

    # Fill in the rest of the chart
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            min_cost = 0
            min_align = CRT_ALIGN
            if hyp[j - 1] == ref[i - 1]:
                min_cost = chart[i - 1][j - 1].cost
                min_align = CRT_ALIGN
            else:
                min_cost = chart[i - 1][j - 1].cost + SUB_COST
                min_align = SUB_ALIGN

            del_cost = chart[i - 1][j].cost + DEL_COST
            if del_cost < min_cost:
                min_cost = del_cost
                min_align = DEL_ALIGN

            ins_cost = chart[i][j - 1].cost + INS_COST
            if ins_cost < min_cost:
                min_cost = ins_cost
                min_align = INS_ALIGN

            chart[i][j].cost = min_cost
            chart[i][j].align = min_align

    crt = sub = ins = det = 0
    i = ref_len
    j = hyp_len
    alignment = []
    while i > 0 or j > 0:
        #if i < 0 or j < 0:
            #break;
        if chart[i][j].align == CRT_ALIGN:
            i -= 1
            j -= 1
            crt += 1
            alignment.append(CRT_ALIGN)
        elif chart[i][j].align == SUB_ALIGN:
            i -= 1
            j -= 1
            sub += 1
            alignment.append(SUB_ALIGN)
        elif chart[i][j].align == DEL_ALIGN:
            i -= 1
            det += 1
            alignment.append(DEL_ALIGN)
        elif chart[i][j].align == INS_ALIGN:
            j -= 1
            ins += 1
            alignment.append(INS_ALIGN)

    total_error = sub + det + ins

    alignment.reverse()
    return (total_error, crt, sub, det, ins, alignment)

def read_sentences(filename, iscn=False):
    map = {}
    tmpdata = [x.split() for x in open(filename).readlines()]
    data = []
    # deal with multiwords
    #print 1982
    for x in tmpdata:
        if x[0] == "0":
            del(x[0])
        if len(x) == 0:
            continue
        if len(x) == 1:
            data.append(x[:])
            continue
        #print x[0]
        s = ' '.join(x[1:])
        s = s.replace('_', ' ')
        data.append(s.split())
        data[-1].insert(0, x[0])

    #print 1983
    for x in data:
        if len(x) == 0:
            continue

        index = getidx(x[0])
        # index = re.sub(r'\.[^\.]*$', '', index)
        if index in map:
            sys.stderr.write('Duplicate index [%s] in file %s\n'
                    % (index, filename))
            continue
            sys.exit(-1)
        # print '\t', index
        if len(x) == 1:
            map[index] = []
        else:
            tmp = x[1:]
            if iscn:
                #print tmp
                tmp = ' '.join(tmp)
                #tmp = tmp.encode('utf8')
                tmp = re.sub(r'\[[\d\.,]+\]', '', tmp)
                tmp = re.sub(r'<[\d\.s]+>\s*\.', '', tmp)
                tmp = re.sub(r'<\w+>\s+', '', tmp)
                tmp = re.sub(r'\s', '', tmp)
            else:
                tmp = [x.lower() for x in tmp]
            map[index] = tmp
    #print 1984
    return map

def usage():
    'Print usage'
    print ('''Usage:
    -r, --ref <ref-file>        reference file.
    -h, --hyp <hyp-file>        hyperthesis file.
    -c, --chinese               CER for Chinese.
    -i, --index                 index file, only use senteces have these index.
    -s, --sentence              print each sentence info.
    --help                      print usage.
    ''')


def get_wer(hypfile, reffile, iscn=False, idxfile=None, printsen=False):
    """
    Calculate word/character error rate

    :param hypfile: hyperthesis file
    :param reffile: reference file
    :param iscn: CER for Chinese
    :param idxfile: index file, only use senteces have these index
    :param printsen: print each sentence info
    :return:

    notation: either or no blanks between Chinese characters has no effect
    """

    if not (reffile and hypfile):
        usage()
        sys.exit(-1)

    ref = read_sentences(reffile, iscn)
    hyp = read_sentences(hypfile, iscn)

    total_ref_len = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    if idxfile:
        idx = [getidx(x) for x in open(idxfile).readlines()]
        tmp = {}
        for x in idx:
            if x in hyp:
                tmp[x] = hyp[x]
            else:
                sys.stderr.write('Warning, empty hyperthesis %s\n' % x)
                tmp[x] = []
        hyp = tmp

    for x in hyp:
        #print x
        if x not in ref:
            continue
            sys.stderr.write('Error, no reference for %s\n' % x)
            sys.exit(-1)

        if len(ref[x]) == 0 or len(hyp[x]) == 0:
            continue

        if iscn:
            ref[x] = ref[x]
            hyp[x] = hyp[x]

        aligninfo = distance(ref[x], hyp[x])
        total_ref_len += len(ref[x])
        total_sub += aligninfo[2]
        total_del += aligninfo[3]
        total_ins += aligninfo[4]

        sen_error = aligninfo[2] + aligninfo[3] + aligninfo[4]
        sen_len = len(ref[x])
        sen_wer = sen_error * 100.0 / sen_len

        # print each sentence's wer
        if printsen:
            print('%s sub %2d del %2d ins %2d ref %d wer %.2f' \
                    % (x, aligninfo[2], aligninfo[3], aligninfo[4],
                            len(ref[x]), sen_wer))

    total_error = total_sub + total_del + total_ins
    wer = total_error * 100.0 / total_ref_len

    # print 'ref len', total_ref_len
    # print 'sub', total_sub
    # print 'del', total_del
    # print 'ins', total_ins
    # print 'wer %.2f' % wer
    #sys.stdout.write('wer %.2f\n' % wer)

    return total_ref_len, total_sub, total_del, total_ins, wer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=None)
    parser.add_argument('--ref', type=str, default=None)

    args = parser.parse_args()

    get_wer(args.hyp, args.ref, iscn=True, idxfile=None, printsen=True)
