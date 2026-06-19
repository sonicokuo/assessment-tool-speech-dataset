# Reference papers (full list)

Compiled from the three redirect literature reviews (2026-06-19). Grouped by topic; arXiv ids (and URLs where non-arXiv).

## A. Token-free 2D grounding (decoupled query heads)

### Root cause of special-token / new-vocabulary degeneration
- Land & Bartolo, *Fishing for Magikarp* (under-trained/glitch tokens), EMNLP 2024 — arXiv:2405.05417 (code: github.com/cohere-ai/magikarp)
- Rumbelow & Watkins, *SolidGoldMagikarp* (2023) — alignmentforum.org/posts/aPeJE8bSo6rAFoLqg/
- Hewitt, *Initializing New Word Embeddings for Pretrained LMs* (2021) — cs.columbia.edu/~johnhew/vocab-expansion.html (code: github.com/john-hewitt/embed-init)
- Mundra et al., vocab-expansion init / convex hull, CoNLL 2024 — arXiv:2407.05841
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 — arXiv:1904.09751
- *Repetition In Repetition Out*, NeurIPS 2023 — arXiv:2310.10226
- *Scaling Laws for Forgetting* — arXiv:2401.05605
- token-embedding degeneration — arXiv:2408.01308

### Decoupled-query precedents (the recommended architecture)
- Carion et al., *DETR* (object queries), ECCV 2020 — arXiv:2005.12872
- Locatello et al., *Slot Attention*, NeurIPS 2020 — arXiv:2006.15055
- Cheng et al., *Mask2Former*, CVPR 2022 — arXiv:2112.01527
- Li et al., *BLIP-2 / Q-Former*, ICML 2023 — arXiv:2301.12597
- Jaegle et al., *Perceiver IO*, ICLR 2022 — arXiv:2107.14795
- Alayrac et al., *Flamingo* (Perceiver Resampler), NeurIPS 2022 — arXiv:2204.14198

### `<SEG>`-embedding-as-query (robust LLM grounding) vs token-emit (brittle)
- Lai et al., *LISA*, CVPR 2024 — arXiv:2308.00692
- Rasheed et al., *GLaMM*, CVPR 2024 — arXiv:2311.03356
- Peng et al., *Kosmos-2* (location tokens) — arXiv:2306.14824
- Chen et al., *Shikra* (numeric coords) — arXiv:2306.15195
- Chen et al., *Pix2Seq* — arXiv:2109.10852
- Xiao et al., *Florence-2* (`<loc_x>`) — arXiv:2311.06242
- Bai et al., *Qwen-VL* (`<box>`) — arXiv:2308.12966
- Liu et al., *Grounding DINO* — arXiv:2303.05499
- *GLIP* — arXiv:2112.03857; *RegionCLIP* — arXiv:2112.09106

### Audio grounding / time-frequency localization (most on-target)
- **Frame-Level Internal Tool Use (Feb 2026) — arXiv:2602.10230** — *the load-bearing precedent*: replaced timestamp-token emission with a decoupled frame head because token-based models "collapse completely" out-of-distribution (= our 16-44% under augmentation).
- *ChunkMOS* (frame-level quality + local-global coupling), Interspeech 2024 — arXiv:2508.10374
- *Text-to-Audio Grounding*, ICASSP 2021 — arXiv:2102.11474
- *MGA-CLAP*, ACM MM 2024 — arXiv:2408.07919
- *AudioSep / LASS* (genuine T-F masks) — arXiv:2308.05037
- Kong et al., weakly-supervised T-F segmentation, TASLP 2019 — arXiv:1804.04715
- *QualiSpeech* (time-IoU eval) — arXiv:2503.20290
- descriptive speech-quality LLM evaluators, ICLR 2025 — arXiv:2501.17202
- *Calibration-Reasoning for Descriptive Speech Quality* (closest competitor, RL temporal localization), 2026 — arXiv:2603.10175

### Attention / attribution faithfulness (for validation; why NOT to headline post-hoc attention)
- *Attention is not Explanation* — arXiv:1902.10186; *...is not not Explanation* — arXiv:1908.04626
- *StreamingLLM* (attention sinks) — arXiv:2309.17453; *Massive Activations* — arXiv:2402.17762
- *Integrated Gradients*, ICML 2017 — arXiv:1703.01365; *Grad-CAM* — arXiv:1610.02391
- *Sanity Checks for Saliency Maps*, NeurIPS 2018 — arXiv:1810.03292
- *ERASER* — arXiv:1911.03429; *SaCo* (ViT faithfulness), CVPR 2024 — arXiv:2404.01415
- guided-attention loss — arXiv:1710.08969; *HINT* — arXiv:1902.03751

## B. Two-stage / alignment for special tokens

### Vocabulary extension & new-token init
- Hewitt mean-init — cs.columbia.edu/~johnhew/vocab-expansion.html
- *FVT* — arXiv:2402.09977; *WECHSEL* — arXiv:2112.06598; *FOCUS* — arXiv:2305.14481
- Mosin *VIPI* — arXiv:2112.14569; *Chinese-LLaMA* — arXiv:2304.08177
- init comparison — arXiv:2407.05841; two-stage 0.01GB vocab expansion — arXiv:2406.11477
- *Token Distillation* (init input + lm_head rows) — arXiv:2505.20133; embedding-variability instability — arXiv:2409.07787

### Two-stage / warmup / staged training
- *BLIP-2* (stage-1 warmup prevents catastrophic forgetting) — arXiv:2301.12597
- *LLaVA* — arXiv:2304.08485; *LLaVA-1.5* — arXiv:2310.03744
- Artetxe et al. (embeddings-only transfer), ACL 2020 — aclanthology.org/2020.acl-main.421/
- *AraLLaMA* (progressive vocab) — arXiv:2412.12310

### Teaching token meaning / soft tokens
- *Prompt Tuning* — arXiv:2104.08691; *Prefix-Tuning* — arXiv:2101.00190; *P-tuning* — arXiv:2103.10385; *PPT* — arXiv:2109.04332
- *Textual Inversion* — arXiv:2208.01618
- definition→embedding: Bahdanau — arXiv:1706.00286; Hill — arXiv:1504.00548; Noraset — arXiv:1612.00394

### Control/structural tokens (how they are successfully taught)
- *Qwen* (ChatML disambiguation) — arXiv:2309.16609; *Qwen3* — arXiv:2505.09388
- *Llama-3 tokenizer* (256 reserved slots above BPE) — github.com/meta-llama/llama3
- *T5* (sentinels) — arXiv:1910.10683; *BERT* — arXiv:1810.04805
- *ViT needs Registers*, ICLR 2024 — arXiv:2309.16588
- *Pause Tokens* (need pretrain+finetune — key counter-evidence), ICLR 2024 — arXiv:2310.02226
- *Quiet-STaR* — arXiv:2403.09629; *DeepSeek-R1* — arXiv:2501.12948; *Toolformer* — arXiv:2302.04761; *Planning Tokens* — arXiv:2310.05707; *CTRL* — arXiv:1909.05858

### Regularization & constrained decoding vs degeneration
- RLHF/KL (Ziegler) — arXiv:1909.08593; KL-as-Bayes (Korbak) — arXiv:2205.11275
- catastrophic-forgetting scale study — arXiv:2308.08747; pretraining-injection scaling law (~1%) — arXiv:2502.06042; mix-review — arXiv:1910.07117
- *SDFT* (self-distillation FT), ACL 2024 — arXiv:2402.13669
- *Language Confusion* (10% in-distribution data fix), EMNLP 2024 — arXiv:2406.20052
- *Grammar-Constrained Decoding*, EMNLP 2023 — arXiv:2305.13971; *Outlines* (FSM) — arXiv:2307.09702; *Grammar-Aligned Decoding*, NeurIPS 2024; *CRANE* (constraint quality tax) — arXiv:2502.09061; KL self-distillation for vocab expansion — arXiv:2508.15807

## C. RL / agentic for faithful generation

### RLVR / GRPO foundations
- DeepSeekMath / *GRPO* — arXiv:2402.03300
- *DeepSeek-R1* (rule-based reward, SFT cold-start) — arXiv:2501.12948
- *Tulu 3* (RLVR named) — arXiv:2411.15124
- *RLOO* ("Back to Basics") — arXiv:2402.14740
- *RAFT* (reward-ranked FT) — arXiv:2304.06767
- *Dr. GRPO* (length-bias fix) — arXiv:2503.20783

### Elicit-vs-teach (the claim to defend re: f0_mean)
- *Does RL Incentivize Reasoning Beyond the Base Model?* — arXiv:2504.13837
- *CoT-Pass@K* — arXiv:2506.14245; *ProRL* — arXiv:2505.24864
- *SFT Memorizes, RL Generalizes* — arXiv:2501.17161; *1-shot RLVR* — arXiv:2504.20571
- *Spurious Rewards* (Qwen-specific caveat) — arXiv:2506.10947

### Reward design / hacking / degeneration
- reward-model over-optimization scaling — arXiv:2210.10760; *Catastrophic Goodhart* — arXiv:2407.14503
- *Let's Verify Step by Step* (process > outcome) — arXiv:2305.20050
- densify sparse reward — arXiv:2510.07242; sparse-reward inflates hallucination — arXiv:2509.03403
- *Unlikelihood Training* — arXiv:1908.04319; *Quark* — arXiv:2205.13636
- RLVR can still be reward-hacked (deterministic-verifier gaming) — arXiv:2604.15149

### Faithfulness / hallucination via RL/DPO
- *FactTune* — arXiv:2311.08401; *Train for Truth* — arXiv:2510.17733
- *PARENTing* (closest numeric-faithfulness precedent) — arXiv:2010.10866
- *CHAIR* — arXiv:1809.02156; *ALOHa* — arXiv:2404.02904

### Agentic / self-correction / verifier-in-loop
- *LLMs Cannot Self-Correct Reasoning Yet* — arXiv:2310.01798; critical survey — arXiv:2406.01297
- *Self-Refine* — arXiv:2303.17651; *Reflexion* — arXiv:2303.11366
- *CRITIC* — arXiv:2305.11738; *PAL* — arXiv:2211.10435; *Self-Debugging* — arXiv:2304.05128
- *SCoRe* (RL to self-correct) — arXiv:2409.12917
- verifier best-of-N (Cobbe) — arXiv:2110.14168; test-time compute — arXiv:2408.03314
- *Woodpecker* (verify-edit cuts caption hallucination) — arXiv:2310.16045
- LLM-as-judge biases — arXiv:2411.16594; multi-agent failure *MAST* — arXiv:2503.13657

### Multimodal / audio RLVR (transfer evidence)
- **R1-AQA (RL>SFT on audio, 38k, Qwen2-Audio-7B; CoT did NOT help) — arXiv:2503.11197**
- *SARI* (SFT→curriculum-GRPO, +16.35%) — arXiv:2504.15900
- *FSA-GRPO* (frozen encoder+adapter, LoRA-only GRPO on audio LLM) — arXiv:2606.02615
- *Visual-RFT* (IoU verifiable reward, beats SFT) — arXiv:2503.01785; *VLM-R1* — arXiv:2504.07615; *Vision-R1* — arXiv:2503.06749
- *SCST* (metric-as-reward; CIDEr) — arXiv:1612.00563; *DiCO* (metric-RL repetition collapse + fix) — arXiv:2408.14547

### Practical recipes / libraries
- TRL GRPOTrainer — huggingface.co/docs/trl/grpo_trainer
- verl — github.com/volcengine/verl
- OpenRLHF — github.com/OpenRLHF/OpenRLHF
- Unsloth GRPO (low-VRAM) — unsloth.ai/blog/grpo
- OpenAI RFT (grader-based) — developers.openai.com/api/docs/guides/reinforcement-fine-tuning
