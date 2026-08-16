#!/bin/bash
# Re-derive every architecture number now written into .claude/CLAUDE.md, from the
# TRAINING LOGS on PSC. Nothing here is taken from a memo, a memory entry, or an agent.
SH_=/ocean/projects/cis260125p/shared
declare -A LOGS=(
  [film-attn_s73]=train_attn_seed73_0725_0255
  [film-attn_s42]=train_attn_seed42_0727_1539
  [film-attn-2L]=train_attn2L_0727_2022
  [film-attn-2L_res]=train_attn2L_resume_0727_2335
  [film-mamba_s73]=train_mamba4_seed73_0727_0145
  [film-mamba_s42]=train_mamba4_seed42_0728_0446
  [film-mamba_s42_res]=train_mamba4_seed42_resume_0729_0058
  [attn-concat_s73]=train_attnconcat_0729_0058
  [concat-only_s73]=train_concatonly_0729_0129
  [T1_compr4]=train_attn_c4_0727_2357
  [fw2_fixedtargets]=train_attn_fw2_0728_0029
)
for k in "${!LOGS[@]}"; do
  f=$SH_/logs/${LOGS[$k]}.log
  if [ ! -f "$f" ]; then echo "$k: LOG MISSING (${LOGS[$k]})"; continue; fi
  vals=$(tr '\r' '\n' < "$f" | grep -aoE 'srcc_robust=[0-9.]+|no srcc_robust improvement \([0-9.]+' \
         | grep -oE '[0-9]+\.[0-9]+' | tr '\n' ' ')
  best=$(echo $vals | tr ' ' '\n' | sort -rn | head -1)
  echo "$k: best=$best   epochs=[ $vals]"
done
