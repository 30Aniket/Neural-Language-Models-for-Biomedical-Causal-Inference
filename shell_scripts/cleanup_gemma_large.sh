#!/usr/bin/env bash
# Inspect and clean up everything the six large-Gemma jobs create.
#
# DRY RUN BY DEFAULT. Nothing is deleted unless you add --yes.
#
#   bash cleanup_gemma_large.sh status            # what exists, how big, fold counts
#   bash cleanup_gemma_large.sh scratch  [--yes]  # mid-fold checkpoints for FINISHED folds
#   bash cleanup_gemma_large.sh caches   [--yes]  # triton / inductor / nv / __pycache__
#   bash cleanup_gemma_large.sh runlogs  [--yes]  # per-run .log files, keeps newest 3 each
#   bash cleanup_gemma_large.sh joblogs  [--yes]  # logs/*.out/.err for jobs no longer queued
#   bash cleanup_gemma_large.sh backups  [--yes]  # *.bak_* left by the installer
#   bash cleanup_gemma_large.sh safe     [--yes]  # all of the above, results untouched
#   bash cleanup_gemma_large.sh hfcache  [--yes]  # the 4-bit weights (~45 GB) -- only when done
#   bash cleanup_gemma_large.sh archive           # tar results for download, deletes nothing
#   bash cleanup_gemma_large.sh reset <model|all> --yes   # DESTROYS predictions, full restart
#
# 'safe' never touches df_res*.csv or train_curve_*.csv -- those are your results.
# Only 'reset' removes them, and only with an explicit model name and --yes.
set -euo pipefail
cd "$(dirname "$0")"

USER="${USER:-$(id -un)}"
MODE="${1:-status}"; shift || true
ARG=""; APPLY=0; FORCE=0
for a in "$@"; do
  case "$a" in
    --yes) APPLY=1 ;;
    --force) FORCE=1 ;;
    *) ARG="$a" ;;
  esac
done

MODELS=(gemma3_12b gemma4_12b gemma4_31b medgemma_4b medgemma_27b mediphi_4b phi4_14b)
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
CONFIG="${CONFIG:-paper}"
TAGS=(10 21 32 43 04)

c_run () { printf '\033[33m%s\033[0m\n' "$*"; }
say () { printf '%s\n' "$*"; }

running_gemma_jobs () {
  command -v squeue >/dev/null 2>&1 || { echo 0; return; }
  squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -c '^gemma' || true
}

size_of () { [[ -e "$1" ]] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }

# rm_list <description> < newline-separated paths on stdin
rm_list () {
  local what="$1" n=0 total=0 p
  local tmp; tmp="$(mktemp)"
  cat > "$tmp"
  n=$(grep -cve '^$' "$tmp" || true)
  if [[ "$n" -eq 0 ]]; then say "  ${what}: nothing to remove"; rm -f "$tmp"; return; fi
  total=$(du -sch $(tr '\n' ' ' < "$tmp") 2>/dev/null | tail -1 | cut -f1 || echo "?")
  if [[ "$APPLY" -eq 1 ]]; then
    while IFS= read -r p; do [[ -n "$p" ]] && rm -rf -- "$p"; done < "$tmp"
    say "  ${what}: removed ${n} item(s), reclaimed ${total}"
  else
    c_run "  ${what}: WOULD remove ${n} item(s), ${total}"
    head -8 "$tmp" | sed 's/^/      /'
    if [[ "$n" -gt 8 ]]; then say "      ... and $((n-8)) more"; fi
  fi
  rm -f "$tmp"
  return 0
}

guard_running () {
  local n; n=$(running_gemma_jobs)
  if [[ "$n" -gt 0 && "$FORCE" -eq 0 ]]; then
    say "refusing: ${n} gemma job(s) still in the queue. Wait, or pass --force." >&2
    exit 1
  fi
}

# --------------------------------------------------------------------------
case "$MODE" in

status)
  say "=== fold completion (df_res<tag>.csv per model x dataset) ==="
  for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
      dt=ailf; [[ "$d" == Tramadol* ]] && dt=tram
      pd="dat/${d}/proc/cross_val_${m}_${CONFIG}_temp"
      have=""
      for t in "${TAGS[@]}"; do [[ -f "${pd}/df_res${t}.csv" ]] && have+="${t} "; done
      n=$(echo $have | wc -w)
      printf '  %-11s %-5s %d/5   %s\n' "$m" "$dt" "$n" "${have:-none}"
    done
  done

  say ""
  say "=== disk usage ==="
  for p in .hf_cache scratch_persist logs outputs .triton_cache .torchinductor_cache \
           .torch .torch_ext .nv_cache .cache; do
    printf '  %-24s %s\n' "$p" "$(size_of "$p")"
  done
  say ""
  printf '  %-24s %s\n' "predictions (all models)" \
      "$(du -sch dat/*/proc/cross_val_*_temp 2>/dev/null | tail -1 | cut -f1 || echo -)"
  say ""
  say "=== job logs ==="
  printf '  logs/gemma*.out logs/medgemma*.out logs/mediphi*.out logs/phi4*.out : %s file(s), %s\n' \
      "$(ls logs/gemma*.out logs/medgemma*.out logs/mediphi*.out logs/phi4*.out 2>/dev/null | wc -l)" \
      "$(du -sch logs/gemma*.out logs/medgemma*.out logs/mediphi*.out logs/phi4*.out 2>/dev/null | tail -1 | cut -f1 || echo -)"
  printf '  logs/gemma*.err logs/medgemma*.err logs/mediphi*.err logs/phi4*.err : %s file(s), %s\n' \
      "$(ls logs/gemma*.err logs/medgemma*.err logs/mediphi*.err logs/phi4*.err 2>/dev/null | wc -l)" \
      "$(du -sch logs/gemma*.err logs/medgemma*.err logs/mediphi*.err logs/phi4*.err 2>/dev/null | tail -1 | cut -f1 || echo -)"
  say ""
  say "=== queue ==="
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "$USER" -o '%.10i %.20j %.2t %.10M %.10L %R' 2>/dev/null || true
  else
    say "  (squeue not available on this host)"
  fi
  say ""
  say "orphaned scratch (fold already has df_res -> checkpoint is dead weight):"
  found=0
  for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do for t in "${TAGS[@]}"; do
    s="scratch_persist/${m}_${CONFIG}_${d}_${t}"
    [[ -d "$s" && -f "dat/${d}/proc/cross_val_${m}_${CONFIG}_temp/df_res${t}.csv" ]] && {
      printf '  %s  (%s)\n' "$s" "$(size_of "$s")"; found=1; }
  done; done; done
  [[ "$found" -eq 0 ]] && say "  none"
  ;;

scratch)
  guard_running
  say "mid-fold checkpoints whose fold already produced predictions:"
  { for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do for t in "${TAGS[@]}"; do
      s="scratch_persist/${m}_${CONFIG}_${d}_${t}"
      [[ -d "$s" && -f "dat/${d}/proc/cross_val_${m}_${CONFIG}_temp/df_res${t}.csv" ]] && echo "$s"
    done; done; done; } | rm_list "finished-fold checkpoints"
  say "note: scratch for UNfinished folds is kept -- that is the mid-fold resume state."
  ;;

caches)
  say "JIT / compile caches (rebuilt automatically on next run):"
  { for p in .triton_cache .torchinductor_cache .nv_cache .torch_ext; do
      [[ -d "$p" ]] && echo "$p"
    done
    find src -type d -name __pycache__ 2>/dev/null
  } | rm_list "caches"
  ;;

runlogs)
  say "per-run python .log files, keeping the newest 3 per model/dataset:"
  { for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
      dir="outputs/${m}_${CONFIG}/${d}/run_logs"
      [[ -d "$dir" ]] || continue
      ls -1t "${dir}"/*.log 2>/dev/null | tail -n +4
    done; done; } | rm_list "old run logs"
  ;;

joblogs)
  say "logs/*.out and *.err for jobs that are no longer in the queue:"
  live=""
  command -v squeue >/dev/null 2>&1 && live="$(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ')"
  { for f in logs/gemma*.out logs/gemma*.err logs/medgemma*.out logs/medgemma*.err logs/mediphi*.out logs/mediphi*.err logs/phi4*.out logs/phi4*.err; do
      [[ -e "$f" ]] || continue
      jid="$(basename "$f" | sed -E 's/.*_([0-9]+)\.(out|err)$/\1/')"
      [[ " $live " == *" $jid "* ]] || echo "$f"
    done; } | rm_list "finished job logs"
  say "tip: 'archive' first if you want them for the dissertation appendix."
  ;;

backups)
  say "installer backups:"
  { find src slurm -maxdepth 2 -name '*.bak_*' 2>/dev/null; } | rm_list "installer backups"
  ;;

safe)
  say "=== safe cleanup: caches, orphaned scratch, old logs, installer backups ==="
  say "results (df_res*.csv, train_curve_*.csv) are never touched by this mode."
  say ""
  for m in caches scratch runlogs joblogs backups; do
    say "-- $m"
    flags=""
    if [[ "$APPLY" -eq 1 ]]; then flags="$flags --yes"; fi
    if [[ "$FORCE" -eq 1 ]]; then flags="$flags --force"; fi
    bash "$0" "$m" $flags || say "  ($m step reported an error, continuing)"
  done
  ;;

hfcache)
  guard_running
  say "Hugging Face weight cache. Deleting this means re-running prefetch_models.sh"
  say "on the login node before any further job. Only do this when ALL runs are done."
  say "  current size: $(size_of .hf_cache)"
  { [[ -d .hf_cache ]] && echo ".hf_cache"; } | rm_list "HF weight cache"
  ;;

archive)
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT="gemma_results_${STAMP}.tar.gz"
  say "packing predictions, training curves and job logs -> ${OUT}"
  tar czf "$OUT" \
      $(for m in "${MODELS[@]}"; do for d in "${DATASETS[@]}"; do
          p="dat/${d}/proc/cross_val_${m}_${CONFIG}_temp"; [[ -d "$p" ]] && echo "$p"
        done; done) \
      $(for m in "${MODELS[@]}"; do
          p="outputs/${m}_${CONFIG}"; [[ -d "$p" ]] && echo "$p"
        done) \
      $(ls logs/gemma*.out logs/medgemma*.out logs/mediphi*.out logs/phi4*.out logs/gemma*.err logs/medgemma*.err logs/mediphi*.err logs/phi4*.err 2>/dev/null || true) 2>/dev/null
  say "wrote ${OUT} ($(size_of "$OUT"))"
  say "download with:"
  say "  scp ${USER}@blythedata.scrtp.warwick.ac.uk:\$PWD/${OUT} ."
  ;;

reset)
  guard_running
  case "$ARG" in
    gemma3_12b|gemma4_12b|gemma4_31b|medgemma_4b|medgemma_27b|mediphi_4b|phi4_14b) TARGETS=("$ARG") ;;
    all) TARGETS=("${MODELS[@]}") ;;
    *) say "usage: bash cleanup_gemma_large.sh reset <gemma3_12b|gemma4_12b|gemma4_31b|medgemma_4b|medgemma_27b|all> --yes" >&2
       exit 2 ;;
  esac
  say "!! DESTRUCTIVE: removes predictions and training curves for: ${TARGETS[*]}"
  say "!! Every fold will be retrained from scratch. Run 'archive' first if unsure."
  [[ "$APPLY" -eq 1 ]] || say "!! dry run -- add --yes to actually delete"
  say ""
  { for m in "${TARGETS[@]}"; do
      for d in "${DATASETS[@]}"; do
        p="dat/${d}/proc/cross_val_${m}_${CONFIG}_temp"; [[ -d "$p" ]] && echo "$p"
        for t in "${TAGS[@]}"; do
          s="scratch_persist/${m}_${CONFIG}_${d}_${t}"; [[ -d "$s" ]] && echo "$s"
        done
      done
      o="outputs/${m}_${CONFIG}"; [[ -d "$o" ]] && echo "$o"
    done; } | rm_list "results for ${TARGETS[*]}"
  ;;

*)
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
  ;;
esac
