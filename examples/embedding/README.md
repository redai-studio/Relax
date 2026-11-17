# Embedding SFT with Qwen3.5-4B

This example trains a text-pair encoder on two data-parallel GPUs using
`--loss-type sft --task-type embedding`. It uses the standard Qwen3.5-4B
Hugging Face checkpoint and pools the last real token from each sequence.

Start a Ray cluster with two available GPUs and set `RAY_ADDRESS` to its Jobs
HTTP address. Set `MODEL_DIR` to the parent directory of `Qwen3.5-4B`, `DATA_DIR`
to the input directory, and `SAVE_DIR` to a persistent checkpoint root.

```bash
mkdir -p "$DATA_DIR"
cat > "$DATA_DIR/pairs.jsonl" <<'EOF'
{"query":"How do I make espresso?","positive":"Brew espresso by pushing hot water through finely ground coffee."}
{"query":"Where do emperor penguins live?","positive":"Emperor penguins live in Antarctica."}
{"query":"Why does ice float?","positive":"Solid water is less dense than liquid water."}
{"query":"What is a prime number?","positive":"A prime has exactly two positive divisors."}
{"query":"How do solar panels work?","positive":"Photovoltaic cells turn light into electricity."}
{"query":"What is the capital of France?","positive":"Paris is the capital of France."}
{"query":"When does a lunar eclipse occur?","positive":"A lunar eclipse occurs when Earth casts its shadow on the Moon."}
{"query":"How can I keep basil fresh?","positive":"Store basil stems upright in a glass of water."}
EOF
bash examples/embedding/scripts/run-qwen35-4B-2xgpu.sh
```

The default run makes two optimizer updates with four pairs per update. Use a
whole number of batches in the JSONL file. Set `NUM_UPDATES` to change the
training length. Use the same `SAVE_DIR` and `EXP_NAME` to resume from a saved
native checkpoint. Set a new `EXP_NAME` for a fresh run.

The framework keeps each pair together through data-parallel assignment and
uses the positives from the full data-parallel batch as InfoNCE candidates.
Set `--max-tokens-per-gpu` to fit the whole local pair batch in one microbatch.
The supported setup is Megatron-Bridge with TP=PP=CP=1.

Custom datasets can supply `input_ids` and an optional zero-based
`embedding_position` per branch. A custom loss can be passed through
`--custom-loss-function-path`. It receives the local query and positive
tensors along with the keyword arguments `temperature` and `process_group`,
and returns a scalar mean loss.
