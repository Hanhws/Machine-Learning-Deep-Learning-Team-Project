#!/usr/bin/env bash
# Colab 업로드용 tar 만들기 + 검증.
#   bash make_upload_tar.sh <car_seg_split_upload 폴더> [출력 tar 경로]
#
# - COPYFILE_DISABLE=1 : macOS tar 가 ._파일명(AppleDouble) 을 끼워 넣지 않게 한다
# - 만든 뒤 tar 목록에서 이미지·라벨·마스크·COCO 주석 수를 원본 폴더와 비교한다
#   (복사 도중 끊긴 tar 는 오류 없이 앞부분만 읽히는 경우가 있어 반드시 확인)
set -euo pipefail

SRC="${1:?사용법: bash make_upload_tar.sh <car_seg_split_upload 폴더> [출력.tar]}"
SRC="${SRC%/}"
NAME="$(basename "$SRC")"
OUT="${2:-$(dirname "$SRC")/$NAME.tar}"

echo "[tar] $SRC → $OUT"
COPYFILE_DISABLE=1 tar -cf "$OUT" -C "$(dirname "$SRC")" "$NAME"

echo "[tar] 검증 중..."
LIST="$(mktemp)"
tar -tf "$OUT" > "$LIST"
fail=0
for d in images/train images/val images/test labels/train labels/val labels/test masks/train masks/val masks/test; do
  [ -d "$SRC/$d" ] || continue
  want=$(find "$SRC/$d" -type f ! -name '._*' | wc -l | tr -d ' ')
  got=$(grep -c "^$NAME/$d/[^/]" "$LIST" || true)
  status="OK"; [ "$want" = "$got" ] || { status="불일치"; fail=1; }
  printf "  %-14s 폴더 %6s  tar %6s  %s\n" "$d" "$want" "$got" "$status"
done
for s in train val test; do
  if grep -q "^$NAME/annotations/instances_$s.json$" "$LIST"; then echo "  annotations/instances_$s.json OK"
  else echo "  annotations/instances_$s.json 없음"; fail=1; fi
done
n_apple=$(grep -c '/\._' "$LIST" || true)
[ "$n_apple" = "0" ] || { echo "  경고: ._ 파일 $n_apple 개 포함"; fail=1; }
rm -f "$LIST"

ls -lh "$OUT"
if [ "$fail" = "0" ]; then echo "[tar] ✅ 검증 통과 — 구글 드라이브에 업로드하세요"; else echo "[tar] ❌ 검증 실패 — 다시 만드세요"; exit 1; fi
