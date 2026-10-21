#!/usr/bin/env bash
# Download the open-license evaluation corpus (all CC BY 4.0) into
# testdata/corpus/raw/ (gitignored). Resumable and idempotent.
#   AMI Meeting Corpus: far-field Array1-01 channel of selected meetings + manual annotations
#   FLEURS id_id (Indonesian read speech): dev + test
#   MUSAN: noise + music (speech part not extracted)
# Sources publish no per-file checksums, so the first download is recorded in
# MANIFEST.sha256 and later runs verify against it.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=testdata/corpus/raw
mkdir -p "$RAW"/{ami,fleurs,musan}

get() { # url dest
  local url=$1 dest=$2
  if [ -f "$dest.done" ]; then return 0; fi
  echo "fetch $url"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest" && touch "$dest.done"
}

AMI=https://groups.inf.ed.ac.uk/ami
MEETINGS="ES2002a ES2002b ES2003a ES2004a ES2005a ES2006a ES2007a ES2008a ES2009a ES2010a
          IS1000a IS1001a IS1002b IS1003a IS1004a IS1005a IS1006a IS1007a IS1008a IS1009a
          TS3003a TS3004a TS3005a TS3006a"
for m in $MEETINGS; do
  url="$AMI/AMICorpusMirror/amicorpus/$m/audio/$m.Array1-01.wav"
  if curl -sfr 0-0 -o /dev/null "$url"; then get "$url" "$RAW/ami/$m.Array1-01.wav"; else echo "skip $m (not on mirror)"; fi
done
get "$AMI/AMICorpusAnnotations/ami_public_manual_1.6.2.zip" "$RAW/ami/ami_public_manual_1.6.2.zip"
[ -d "$RAW/ami/annotations" ] || (mkdir -p "$RAW/ami/annotations" && cd "$RAW/ami/annotations" && unzip -q ../ami_public_manual_1.6.2.zip 'segments/*' 'corpusResources/*')

HF=https://huggingface.co/datasets/google/fleurs/resolve/main/data/id_id
for split in dev test; do
  get "$HF/$split.tsv" "$RAW/fleurs/$split.tsv"
  get "$HF/audio/$split.tar.gz" "$RAW/fleurs/$split.tar.gz"
  [ -d "$RAW/fleurs/$split" ] || tar -xzf "$RAW/fleurs/$split.tar.gz" -C "$RAW/fleurs"
done

# MUSAN (10.4 GB): one openslr mirror serves ~200 KB/s, so fetch in parallel from
# several official mirrors with aria2 (resumable via its .aria2 control file).
if [ ! -f "$RAW/musan/musan.tar.gz.done" ]; then
  ARIA="aria2c --file-allocation=none -x 16 -s 16 -k 4M --summary-interval=60 --console-log-level=warn \
    -d $RAW/musan -o musan.tar.gz \
    https://openslr.elda.org/resources/17/musan.tar.gz \
    https://www.openslr.org/resources/17/musan.tar.gz \
    https://openslr.trmal.net/resources/17/musan.tar.gz"
  if command -v aria2c >/dev/null; then $ARIA; else nix-shell -p aria2 --run "$ARIA"; fi
  [ -f "$RAW/musan/musan.tar.gz.aria2" ] || touch "$RAW/musan/musan.tar.gz.done"
fi
[ -d "$RAW/musan/musan/noise" ] || tar -xzf "$RAW/musan/musan.tar.gz" -C "$RAW/musan" musan/noise musan/music

# Record or verify checksums of the downloaded archives/files.
cd "$RAW"
files=$(find ami fleurs musan -maxdepth 1 -type f \( -name '*.wav' -o -name '*.zip' -o -name '*.tar.gz' -o -name '*.tsv' \) | sort)
if [ -f MANIFEST.sha256 ]; then sha256sum -c --quiet MANIFEST.sha256 && echo "corpus verified";
else sha256sum $files > MANIFEST.sha256 && echo "corpus manifest written"; fi
du -sh ami fleurs musan
