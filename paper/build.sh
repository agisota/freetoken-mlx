#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
paper_build_tmp=$(mktemp -d)
trap 'rm -rf "$paper_build_tmp"' EXIT

tectonic_bin=${TECTONIC:-tectonic}
if ! command -v "$tectonic_bin" >/dev/null 2>&1 && [[ ! -x "$tectonic_bin" ]]; then
  echo "Tectonic is required: https://tectonic-typesetting.github.io/" >&2
  exit 1
fi

curl -L --fail --silent --show-error \
  https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip \
  -o "$paper_build_tmp/neurips2026.zip"
unzip -q "$paper_build_tmp/neurips2026.zip" neurips_2026.sty \
  -d "$paper_build_tmp"
cp "$repo_root/paper/main.tex" "$repo_root/paper/references.bib" \
  "$paper_build_tmp/"
mkdir -p "$paper_build_tmp/figures"
cp "$repo_root/paper/figures/memory_throughput_curve.pdf" \
  "$paper_build_tmp/figures/"

mkdir -p "$repo_root/output/pdf"
(
  cd "$paper_build_tmp"
  "$tectonic_bin" --keep-logs main.tex
)
mv "$paper_build_tmp/main.pdf" \
  "$repo_root/output/pdf/freetoken_mlx_ml4sys2026.pdf"
echo "$repo_root/output/pdf/freetoken_mlx_ml4sys2026.pdf"
