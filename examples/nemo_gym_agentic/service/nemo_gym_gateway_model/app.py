# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""NeMo Gym model-server plugin that hosts the shared Relax Gateway."""

from __future__ import annotations

from fastapi import FastAPI
from nemo_gym.base_responses_api_model import BaseResponsesAPIModel, BaseResponsesAPIModelConfig
from nemo_gym.server_utils import SimpleServer, is_nemo_gym_fastapi_entrypoint

from examples.nemo_gym_agentic.service.app import create_app
from examples.nemo_gym_agentic.service.config import GatewaySettings, validate_nemo_gym_graph


class RelaxGatewayModelConfig(BaseResponsesAPIModelConfig):
    pass


class RelaxGatewayModel(BaseResponsesAPIModel, SimpleServer):
    """One long-lived model server that also owns the trial registry."""

    config: RelaxGatewayModelConfig

    def setup_webserver(self) -> FastAPI:
        if self.config.num_workers not in {None, 1}:
            raise RuntimeError("Relax Gateway model requires exactly one process worker")
        settings = GatewaySettings.from_env()
        validate_nemo_gym_graph(
            self.server_client.global_config_dict,
            gateway_name=self.config.name,
            settings=settings,
        )
        return create_app(settings=settings)


if is_nemo_gym_fastapi_entrypoint(__file__):
    app = RelaxGatewayModel.run_webserver()


if __name__ == "__main__":
    RelaxGatewayModel.run_webserver()
