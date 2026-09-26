# Relax Gateway Model

This NeMo Gym model-server plugin hosts the shared Gateway and callback bridge
inside the long-lived Gym server graph. NeMo Gym calls it through
`/ng-rollout/<id>/...`; the Gateway resolves that opaque capability to one
in-memory Relax endpoint and Bearer token.

Copy this directory into the pinned NeMo Gym source tree as
`responses_api_models/relax_gateway_model` while building the Gym image.
Start the Gym graph with `observability_enabled=true` and a fixed, externally
reachable host/port for this model server.

The selected agent must preserve NeMo Gym's rollout prefix. The Gateway injects
`_ng_task_index` and `_ng_rollout_index` into `/run`.

Unprefixed model calls are rejected because they cannot be safely associated
with a Relax session.
