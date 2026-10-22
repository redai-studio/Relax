# Community QSA attention sources

These optional attention backends are vendored so Relax can select them without
changing the pinned Megatron-LM checkout. The original notices remain in each
file; a Relax copyright line and provenance comment were prepended, with no
kernel logic changes.

| Backend        | Upstream commit                                                                                                                                                                                                                     | Files                                                                                                                    | Original SHA-256                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| `pr7553`       | [BestJuly/Megatron-LM `f42ee1a80f0c1fbf52835d5d47edf4a53cebf8c5`](https://github.com/BestJuly/Megatron-LM/blob/f42ee1a80f0c1fbf52835d5d47edf4a53cebf8c5/megatron/core/transformer/experimental_attention_variant/ops/triton_qsa.py) | `pr_triton_qsa.py`                                                                                                       | `6fc2e5ea83b176e502c23d93804bd59bab3bc5574297cd59c793b42617d72d4b` |
| `tilelang7060` | [AllenFeiZZ/Megatron-LM `a904109c3a67a0c0535262ce4eb02bd83d933d6b`](https://github.com/AllenFeiZZ/Megatron-LM/tree/a904109c3a67a0c0535262ce4eb02bd83d933d6b/megatron/core/transformer/experimental_attention_variant/ops)           | `qsa_tilelang.py`, `qsa_sparse_gqa_fwd.py`, `qsa_sparse_gqa_bwd.py`, `qsa_tilelang_utils.py`, `qsa_selector_counters.py` | Listed below                                                       |

Original TileLang SHA-256 values, in the file order above:

```text
63a47af86cb72dd5f015002aae8b562a05a521961a8f42a8e5b38a6b577682fe
c4eb523518f70c1e77bb689f43e595ca40b180264ccc2b4d4963b6b38e173c05
a3b1daf873c921fad79e3118eb75289781390dc30a592c741e40afa7671cd5b2
0b159e9f445d9e7be196d391b12c7dd446994d410a306537408168837c1acf7b
b3ceb4617e143f3fb3ae76b5e32055956220425dd015bb4715703fa440c41535
```

`pr7553` uses Triton, already present with the training PyTorch environment.
`tilelang7060` requires TileLang; the training stage in `docker/Dockerfile`
already installs it. The GPU validation environment used TileLang 0.1.11.

The Relax patch uses Megatron's existing `id_sparse` selector and Stage-2 loss.
Neither community implementation's standalone indexer or selector is enabled by
this integration. Packed THD routes are converted from document-local blocks to
physical token positions; when Stage-2 metadata is absent, the Triton route uses
physical token offsets as block IDs. CP uses Megatron's existing full-sequence
reconstruction before calling these backends. The existing FlexAttention/fused
backend remains the default when no community backend is selected.
