#!/bin/bash

. ./path.sh

feat="feats_cmvn.scp"
phone_dic='phone.vocab'
oov="<unk>"

. utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: $0 <data-dir>";
    exit 1;
fi

dir=$1
# token_dic=$3
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
rm -f ${tmpdir}/*.scp

echo $tmpdir

# input, which is not necessary for decoding mode, and make it as an optio
utils/data/get_utt2num_frames.sh ${dir} &> /dev/null
cp ${dir}/utt2num_frames ${tmpdir}/ilen.scp
cp ${dir}/${feat} ${tmpdir}/feats.scp
feat-to-dim scp:${tmpdir}/feats.scp ark,t:${tmpdir}/idim.scp &> /dev/null

cp ${dir}/trans.phone ${tmpdir}/phone.scp
cp ${dir}/text ${tmpdir}/text
cp ${dir}/${feat} ${tmpdir}/feat.scp

# cat ${tmpdir}/phone.scp | utils/token2int.pl --map-oov ${oov} -f 2- ${phone_dic} > ${tmpdir}/phone_id.scp
# cat ${tmpdir}/token.scp | utils/token2int.pl --map-oov ${oov} -f 2- ${token_dic} > ${tmpdir}/token_id.scp
python vocab.py -m 'look_up' --vocab ${phone_dic} --trans ${tmpdir}/phone.scp --output ${tmpdir}/phone_id.scp

echo "get phone_len";
# cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/olen.scp
cat ${tmpdir}/phone_id.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/phone_len.scp

echo "get phone size";
# +1 comes from 0-based dictionary
vocsize=`wc -l ${phone_dic} | awk '{print $1}'`
odim=`echo "$vocsize + 1" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${tmpdir}/phone.scp > ${tmpdir}/phone_size.scp

rm -f ${tmpdir}/*.json
for x in ${dir}/text ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | python scp2json.py --key ${k} > ${tmpdir}/${k}.json
done

echo "merging json";
python mergejson.py ${tmpdir}/*.json > ${dir}/data.json

rm -fr ${tmpdir}
