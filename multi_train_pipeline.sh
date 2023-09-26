# AMPERE
wget https://xinyuhua.github.io/resources/naacl2019/naacl19_dataset.zip
unzip naacl19_dataset.zip
mkdir -p data/raw/ampere
mv dataset/iclr_anno_final/* data/raw/ampere/
rm naacl19_dataset.zip
rm -r dataset/

# DISAPERE
wget https://github.com/nnkennard/DISAPERE/raw/main/DISAPERE.zip
unzip DISAPERE.zip
mkdir -p data/raw/disapere
mv DISAPERE/final_dataset/* data/raw/disapere/
rm -r DISAPERE*

#rev_adv
ID=1nJdljy468roUcKLbVwWUhMs7teirah75
FILENAME=dataset.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nJdljy468roUcKLbVwWUhMs7teirah75' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nJdljy468roUcKLbVwWUhMs7teirah75" -O $FILENAME && rm -rf /tmp/cookies.txt
unzip dataset.zip
mkdir -p data/raw/revadv
rm -r dataset/ICLR*
rm -r dataset/NIPS*
mv dataset/* data/raw/revadv/
rm dataset.zip
rm -r dataset/

mv who_wins/data/raw/revadv/review_with_aspect.jsonl.gz data/raw/revadv/
gzip -d data/raw/revadv/review_with_aspect.jsonl.gz
mkdir data/raw/revadv/all
python valence/preprocess_data.py
