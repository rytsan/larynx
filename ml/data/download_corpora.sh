#!/usr/bin/env bash
# Training corpora (§9.1) into $LARYNX_DATA (default: data/, gitignored). Resumable (curl -C -).
# The archive is kept next to its extraction; a re-run skips corpora already extracted.
#   ml/data/download_corpora.sh [vocalset] [vctk]      (no args = all scripted corpora)
# Not scripted (manual, see ml/data/credits.md): LibriTTS-R, JVS-MuSiC, PTDB-TUG.
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
    unzip -q "$file" -d "$DATA/$name.tmp" && mv "$DATA/$name.tmp" "$DATA/$name"
}

for c in ${@:-vocalset vctk}; do
    case $c in
        vocalset) fetch vocalset https://zenodo.org/api/records/1442513/files/VocalSet1-2.zip/content \
                        c5e5efab412637fc94972b93c343a2f0 VocalSet1-2.zip ;;
        vctk) fetch vctk https://datashare.ed.ac.uk/bitstreams/535f4286-e54c-4038-838c-a02285e32cb2/download \
                    8a6ba2946b36fcbef0212cad601f4bfa VCTK-Corpus-0.92.zip ;;
        *) echo "unknown corpus $c"; exit 2 ;;
    esac
done
