#!/bin/bash

. ./path.sh

lang=""
feat="" # feat.scp
oov="<unk>"
verbose=0

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    exit 1;
fi

dir=$1
phone_dic=$2
# token_dic=$3
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

# input, which is not necessary for decoding mode, and make it as an option
if [ ! -z ${feat} ]; then
    if [ ${verbose} -eq 0 ]; then
        utils/data/get_utt2num_frames.sh ${dir} &> /dev/null
        cp ${dir}/utt2num_frames ${tmpdir}/ilen.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp &> /dev/null
    else
        utils/data/get_utt2num_frames.sh ${dir}
        cp ${dir}/utt2num_frames ${tmpdir}/ilen.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir}/idim.scp
    fi
fi

# python text2token.py -s 1 -n 1 ${dir}/text > ${tmpdir}/token.scp
cp ${dir}/text2phone.txt.nosil ${tmpdir}/phone.scp
# cp ${dir}/text ${tmpdir}/token.scp

cat ${tmpdir}/phone.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${phone_dic} > ${tmpdir}/phone_id.scp
# cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${token_dic} > ${tmpdir}/token_id.scp

# cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
cat ${tmpdir}/phone_id.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
# +1 comes from 0-based dictionary
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 1" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/odim.scp

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/lang.scp
fi
# feats
cat ${feat} > ${tmpdir}/feat.scp

rm -f ${tmpdir}/*.json
for x in ${dir}/text ${dir}/utt2spk ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | python scp2json.py --key ${k} > ${tmpdir}/${k}.json
done
python mergejson.py --verbose ${verbose} ${tmpdir}/*.json

rm -fr ${tmpdir}
