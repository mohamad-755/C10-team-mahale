#!/bin/bash
# Step 1: Download + extract clean Swahili Wikipedia article text
#
# Run this on your own machine (not in a sandbox) since it needs to reach
# dumps.wikimedia.org, which isn't reachable from this environment.
#
# Requirements: python3, pip, ~2-3 GB free disk space, a stable connection
# (the compressed dump is roughly a few hundred MB; swap "sw" for another
# language's wiki code later, e.g. "zu" for Zulu, "yo" for Yoruba)

set -e  # stop immediately if any step fails

LANG_CODE="sw"                      # ISO code: sw=Swahili, zu=Zulu, yo=Yoruba, ig=Igbo, ha=Hausa, am=Amharic, ny=Chichewa, luo=Dholuo
WIKI="${LANG_CODE}wiki"
DUMP_DATE="latest"                  # Wikimedia always keeps a "latest" symlink to the newest monthly dump
DUMP_FILE="${WIKI}-${DUMP_DATE}-pages-articles.xml.bz2"
DUMP_URL="https://dumps.wikimedia.org/${WIKI}/${DUMP_DATE}/${DUMP_FILE}"

OUT_DIR="./data/${LANG_CODE}_wikipedia"
RAW_DIR="${OUT_DIR}/raw"
EXTRACTED_DIR="${OUT_DIR}/extracted"

mkdir -p "$RAW_DIR" "$EXTRACTED_DIR"

echo "=== Step 1a: Installing wikiextractor ==="
# wikiextractor knows how to parse Wikipedia's XML dump format and strip
# out markup (templates, tables, infoboxes) to leave clean readable text.
pip install wikiextractor --break-system-packages --quiet

echo "=== Step 1b: Downloading dump for ${WIKI} ==="
echo "URL: ${DUMP_URL}"
curl -L --fail -o "${RAW_DIR}/${DUMP_FILE}" "${DUMP_URL}"

echo "=== Step 1c: Extracting clean article text (excluding non-article namespaces) ==="
# --no-templates speeds this up since we don't need infobox templates expanded
# --json outputs one JSON object per article: {"id", "title", "text"}
# wikiextractor automatically skips non-article namespaces (Category:/Jamii:,
# Talk:/Majadiliano:, User:/Mtumiaji:, etc.) - this is exactly the filtering
# we identified as necessary after seeing "Jamii:" category pages in search results.
python3 -m wikiextractor.WikiExtractor \
    "${RAW_DIR}/${DUMP_FILE}" \
    --output "${EXTRACTED_DIR}" \
    --json \
    --no-templates \
    --processes 4

echo "=== Done ==="
echo "Raw dump saved to:      ${RAW_DIR}/${DUMP_FILE}"
echo "Clean extracted text:   ${EXTRACTED_DIR}/ (nested folders of .json files, one article per line)"
echo ""
echo "Next: run 02_spot_check_sample.py against ${EXTRACTED_DIR} to pull your 20-document sample."
