#!/usr/bin/env bash
# Training corpora (§9.1) into $LARYNX_DATA (default: data/, gitignored). Resumable (curl -C -).
# The archive is kept next to its extraction; a re-run skips corpora already extracted.
#   ml/data/download_corpora.sh [vocalset] [vctk] [libritts_r] [ptdb]   (no args = all)
# JVS-MuSiC is not scriptable (no stable public URL): extract it by hand into $DATA/jvs_music.
set -euo pipefail
DATA=${LARYNX_DATA:-data}
mkdir -p "$DATA"

fetch() { # name url md5 archive
    local name=$1 url=$2 md5=$3 file=$DATA/$4
    [ -d "$DATA/$name" ] && return
    echo "== $name"
    curl -L --fail --retry 5 -C - -o "$file" "$url"
    echo "$md5  $file" | md5sum -c -
    rm -rf "$DATA/$name.tmp" && mkdir "$DATA/$name.tmp"
    case $file in
        *.zip) python3 -m zipfile -e "$file" "$DATA/$name.tmp" ;; # Info-ZIP rejects big zip64 as a "zip bomb"
        *.tar.gz) tar -xzf "$file" -C "$DATA/$name.tmp" ;;
    esac
    mv "$DATA/$name.tmp" "$DATA/$name"
}

for c in ${@:-vocalset vctk libritts_r ptdb}; do
    case $c in
        # v1.1: the 5.7 GB v1.2 zip has >4 GB offsets without zip64 records and does not extract cleanly
        vocalset) fetch vocalset https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content \
                        0c09396242f946e7111ad7d8fc649b81 VocalSet11.zip ;;
        vctk) fetch vctk https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download \
                    8a6ba2946b36fcbef0212cad601f4bfa VCTK-Corpus-0.92.zip ;;
        libritts_r) fetch libritts_r https://www.openslr.org/resources/141/train_clean_100.tar.gz \
                          6df668d8f5f33e70876bfa33862ad02b libritts_r_train_clean_100.tar.gz ;;
        ptdb) fetch ptdb https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip \
                    e590798aa0befaf7b458d3c38cda4030 PTDB-TUG.zip ;;
        *) echo "unknown corpus $c"; exit 2 ;;
    esac
done
