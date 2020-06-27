# -- IMPORTANT
dir=$1

echo "make json files"
for data in train test dev; do
    ./data2json.sh --feat ${dir}/${data}/feats.scp ${dir}/${data} phones.txt > ${dir}/${data}/data.json
done
