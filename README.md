<div align="center">

# When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models

</div>

### Overview
- **Goal**: Assess how irrelevant audio interferes with textual reasoning across audio-language models.
- **Benchmarks**: GSM8K, MMLU, ARC-Challenge.
- **Settings**:
  - **text_bench**: text-only baseline
  - **text_bench_interference**: text + irrelevant audio

Public datasets used in the paper are on the Hugging Face Hub:
- GSM8K (interference): `https://huggingface.co/datasets/lca0503/audio_interference_gsm8k`
- MMLU (interference): `https://huggingface.co/datasets/lca0503/audio_interference_mmlu`
- ARC-Challenge (interference): `https://huggingface.co/datasets/lca0503/audio_interference_arc_challenge`


### Setup
1) Clone and enter repo
```bash
git clone https://github.com/lca0503/AudioInterference.git
cd AudioInterference
```
2) Install dependencies
```bash
pip install -r requirements.txt
```


### Build datasets (ptional)
You can construct the interference datasets by pairing each test question with a random audio file (.wav). Provide a folder of audio files and a target Hub repo name.

- GSM8K
```bash
python build_dataset/gsm8k.py \
  --audio_path /path/to/wavs \
  --repo_name your-username/audio_interference_gsm8k \
  --seed 0
```
- MMLU
```bash
python build_dataset/mmlu.py \
  --audio_path /path/to/wavs \
  --repo_name your-username/audio_interference_mmlu \
  --seed 0
```
- ARC-Challenge
```bash
python build_dataset/arc_challenge.py \
  --audio_path /path/to/wavs \
  --repo_name your-username/audio_interference_arc_challenge \
  --seed 0
```

Utility generators (optional):
- Gaussian noise
```bash
python build_dataset/generate_noise.py \
  --output_dir ./noise_wavs \
  --num_noise 1000 \
  --sampling_rate 16000 \
  --duration 5 \
  --sigma 0.01 \
  --seed 0
```
- Silence
```bash
python build_dataset/generate_silence.py \
  --output_dir ./silence_wavs \
  --num_silence 1000 \
  --sampling_rate 16000 \
  --duration 5
```
- FSD50K
Download from `https://zenodo.org/records/4060432`


### Inference
Common arguments
- **--task_id**: one of `audio_interference_gsm8k`, `audio_interference_mmlu`, `audio_interference_arc_challenge`
- **--task_split**: one of `silence`, `noise`, `fsd`, etc. 
- **--task_type**: `text_bench` or `text_bench_interference`
- **--mitigate_prompt**: add this flag to use the mitigation prompt
- **--output_path**: JSONL path to save results

Results are saved as JSONL with fields like `subject`, `task`, `prompt`, `query`, `choices`, `response`, `answer`.

Examples:
- Qwen2.5-Omni (via vLLM)
```bash
python inference_qwen25omni.py \
  --task_id audio_interference_arc_challenge \
  --task_split silence \
  --task_type text_bench_interference \
  --model_id Qwen/Qwen2.5-Omni-7B \
  --output_path outputs/qwen25omni/silence_arc_challenge.jsonl \
  --temperature 0 \
  --seed 0
```
- Phi-4-multimodal-instruct (via vLLM + LoRA)
```bash
python inference_phi4mm.py \
  --task_id audio_interference_gsm8k \
  --task_split noise \
  --task_type text_bench_interference \
  --model_id microsoft/Phi-4-multimodal-instruct \
  --output_path outputs/phi4mm/noise_gsm8k.jsonl \
  --temperature 0 \
  --seed 0
```
- Voxtral (Mistral-format via vLLM)
```bash
python inference_voxtral.py \
  --task_id audio_interference_mmlu \
  --task_split fsd \
  --task_type text_bench_interference \
  --model_id mistralai/Voxtral-Mini-3B-2507 \
  --output_path outputs/voxtralmini/fsd_mmlu.jsonl \
  --seed 0
```
- DeSTA2.5 (via Transformers and `https://github.com/kehanlu/DeSTA2.5-Audio`)
```bash
python inference_desta.py \
  --task_id audio_interference_mmlu \
  --task_split fsd \
  --task_type text_bench_interference \
  --model_id DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B \
  --output_path outputs/desta25/fsd_mmlu.jsonl \
  --seed 0
```

Notes
- Adjust CUDA/vLLM configs per your hardware (GPU memory, max_model_len, etc.).
- For text-only baselines, set `--task_type text_bench`.


### Evaluation
##### Accuracy
Compute accuracy from JSONL results.

Example:
```bash
python evaluate.py \
  --input_path outputs/qwen25omni/silence_arc_challenge.jsonl \
  --task_id arc
python evaluate.py \
  --input_path outputs/phi4mm/noise_gsm8k.jsonl \
  --task_id gsm8k
```

Self-consistency (majority vote over multiple responses):
```bash
python evaluate.py --input_path your.jsonl --task_id mmlu --scs
```
When using self-consistency, ensure each sample's `response` in the JSONL is a list of strings.

##### Influence rate (compare vs. text_bench)
Compute inconsistency rate (IR) between an interference run and the corresponding text-only baseline.

```bash
python influence_rate.py \
  --input_path outputs/qwen25omni/silence_mmlu.jsonl \
  --target_path outputs/qwen25omni/mmlu.jsonl \
  --task_id mmlu
python influence_rate.py \
  --input_path outputs/phi4mm/noise_gsm8k.jsonl \
  --target_path outputs/phi4mm/gsm8k.jsonl \
  --task_id gsm8k
```

Self-consistency (majority vote over multiple responses):
```bash
python influence_rate.py --input_path interfered.jsonl --target_path text.jsonl --task_id mmlu --scs
```


### Citation
If you find our code or models helpful, please consider citing our paper using the following BibTeX:
```
TBD
```
