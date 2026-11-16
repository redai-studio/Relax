# ART integration provenance

SenseNova model architecture and processing are adapted from DiffSynth-Studio,
with source attribution retained in the existing model files.

ART formulas follow [DiffusionART](https://github.com/vanzll/DiffusionART),
revision `58e2052`, under Apache-2.0. Relax owns the distributed replay,
checkpoint, engine transport and scheduling integration.

The TextPecker sentence/substring helpers in
`relax/engine/rewards/_textpecker_ff.py` follow Flow-Factory revision
`6d5b4710f7f1d504fe8f09c6e506ba9f21644e30` (Apache-2.0), preserving its license
header. LPIPS and AlexNet weights are separately provisioned artifacts under
their respective licenses. Runtime transport and invalid-candidate handling are
Relax integration code.
