#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--multi', '-m', type=int,
                        help='Test the json file for multiple input/output', default=0)
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # make intersection set for utterance keys
    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info('new json has ' + str(len(intersec_ks)) + ' utterances')

    old_dic = dict()
    for k in intersec_ks:
        v = js[0]['utts'][k]
        for j in js[1:]:
            v.update(j['utts'][k])
        old_dic[k] = v

    new_dic = dict()
    for id in old_dic:
        dic = old_dic[id]

        in_dic = {}
        try:
            in_dic['shape'] = (int(dic['ilen']), int(dic['idim']))
        except:
            pass
        in_dic['name'] = 'input1'
        in_dic['feat'] = dic['feat']

        out_dic = {}
        out_dic['name'] = 'phone'
        out_dic['shape'] = (int(dic['phone_len']), int(dic['phone_size']))

        try:
            out_dic['phone'] = dic['phone']
            out_dic['phone_id'] = dic['phone_id']
        except:
            pass

        try:
            out_dic['text'] = dic['text']
        except:
            pass

        try:
            out_dic['token'] = dic['token']
            out_dic['token_id'] = dic['token_id']
        except:
            pass

        new_dic[id] = {'input':[in_dic], 'output':[out_dic]}

    jsonstring = json.dumps({'utts': new_dic}, indent=2, ensure_ascii=False, sort_keys=True)

    print(jsonstring)
