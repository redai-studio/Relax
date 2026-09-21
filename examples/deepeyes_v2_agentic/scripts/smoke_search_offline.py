# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, NoReturn
from unittest.mock import Mock, patch

import httpx
import openai
import yaml


EXAMPLE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
sys.path[:0] = [str(EXAMPLE_DIR), str(REPO_ROOT)]

from app import agent, env_deepeyes_v2, search_http, search_utils  # noqa: E402
from app.prompt import UNIFIED_SYSTEM_PROMPT  # noqa: E402
from app.search_config import SEARCH_CONFIG_ENV, load_search_config  # noqa: E402


QUERY = "DeepEyes-V2 离线搜索验证"
SEARCH_TEXT = "<tool_call>" + json.dumps({"name": "search", "arguments": {"query": QUERY}}) + "</tool_call>"
SCENARIOS = ("mock", "retriever-error", "external-error")


def require(condition: bool, reason: str) -> None:
    """在验证条件不满足时立即抛出带固定原因的 RuntimeError."""

    if not condition:
        raise RuntimeError(reason)


def write_json(path: Path, payload: Any) -> None:
    """将 smoke 数据写入 UTF-8 JSON，覆盖同名文件，父目录需已经存在."""

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def smoke_environment(scenario: str, directory: Path) -> dict[str, str]:
    """为已验证的 scenario 生成受控环境，错误场景在 directory 写入搜索配置."""

    # 认证值仅供本进程中的受控 transport 使用。
    values = {
        "OPENAI_API_KEY": "offline-model-token",
        "OPENAI_BASE_URL": "https://model.example.test/v1",
        "OPENAI_MODEL": "offline-controlled-model",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if "TMPDIR" in os.environ:
        values["TMPDIR"] = os.environ["TMPDIR"]
    if scenario == "mock":
        return values
    backend = "retriever" if scenario == "retriever-error" else "external"
    config: dict[str, Any] = {"backend": backend}
    if backend == "external":
        config = yaml.safe_load((EXAMPLE_DIR / "search_config.brave.yaml").read_text(encoding="utf-8"))
        config["auth"]["env"] = "SMOKE_SEARCH_TOKEN"
        values["SMOKE_SEARCH_TOKEN"] = "offline-search-token"
    config.update(
        endpoint=f"https://{backend}.example.test/search",
        max_retries=2,
        retry_delay_s=0.0,
        retry_max_delay_s=0.0,
        trust_env=False,
    )
    config_path = directory / "search_config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    values[SEARCH_CONFIG_ENV] = str(config_path)
    return values


class OfflineSmoke:
    """记录受控模型驱动的 agent 搜索流程，验证消息、重试和资源关闭行为."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.requests: list[dict[str, Any]] = []
        self.search_results: list[Any] = []
        self.search_requests: list[httpx.Request] = []
        self.search_responses: list[httpx.Response] = []
        self.model_clients: list[openai.AsyncOpenAI] = []
        self.search_clients: list[httpx.Client] = []
        self.closed_environments: list[env_deepeyes_v2.DeepEyesV2Env] = []
        self.events: list[str] = []
        self.network_attempts = 0
        self.real_model_client = openai.AsyncOpenAI
        self.answer = "离线搜索验证完成。" if scenario == "mock" else "搜索暂时不可用，已完成后续回答。"

    def deny_network(self, *args: Any, **kwargs: Any) -> NoReturn:
        """记录同步网络访问尝试并立即终止该访问."""

        self.network_attempts += 1
        raise RuntimeError("offline_network_access_forbidden")

    async def deny_async_network(self, *args: Any, **kwargs: Any) -> NoReturn:
        """通过相同检查拒绝异步网络访问."""

        self.deny_network(*args, **kwargs)

    def model_response(self, request: httpx.Request) -> httpx.Response:
        """提供受控模型响应，依次要求搜索和给出最终答案."""

        require(request.method == "POST" and request.url.path == "/v1/chat/completions", "invalid_model_request")
        self.requests.append(json.loads(request.content))
        self.events.append("model")
        require(len(self.requests) <= 2, "unexpected_model_request")
        text = SEARCH_TEXT if len(self.requests) == 1 else f"<answer>{self.answer}</answer>"
        return httpx.Response(
            200,
            json={
                "id": f"offline-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-controlled-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def model_client(self, **kwargs: Any) -> openai.AsyncOpenAI:
        """创建使用内存 transport 的 OpenAI 客户端并登记关闭检查对象."""

        client = self.real_model_client(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.model_response), trust_env=False),
        )
        self.model_clients.append(client)
        return client

    def search_response(self, request: httpx.Request) -> httpx.Response:
        """记录搜索请求并返回受控 503 响应，供重试及错误恢复验证使用."""

        self.search_requests.append(request)
        response = httpx.Response(503, json={"error": "offline-unavailable"})
        self.search_responses.append(response)
        return response

    def search_client(self, config: search_http.HttpSearchConfig) -> httpx.Client:
        """为错误场景创建内存 HTTP 客户端，并拒绝 mock 创建网络客户端."""

        require(self.scenario != "mock", "mock_created_search_http_client")
        client = httpx.Client(
            transport=httpx.MockTransport(self.search_response),
            timeout=httpx.Timeout(config.timeout_s),
            follow_redirects=False,
            trust_env=False,
        )
        self.search_clients.append(client)
        return client

    def search(self, query: str, size: int | None = None) -> Any:
        """记录事件和结果，同时调用示例的真实搜索入口."""

        self.events.append("search")
        require(query == QUERY and size is None, "unexpected_search_arguments")
        result = search_utils.search(query, size=size)
        self.search_results.append(result)
        return result

    async def close_model_clients(self) -> None:
        """关闭运行器创建的全部模型客户端."""

        for client in self.model_clients:
            await client.close()

    def verify(self, initial: list[dict[str, Any]], output: dict[str, Any]) -> dict[str, Any]:
        """核查输入历史、工具观察、最终答案及资源状态，返回可保存的验证报告.

        条件不满足时抛出 RuntimeError，报告包含当前 scenario 的请求次数和错误标记.
        """

        failure = self.scenario != "mock"
        require(self.network_attempts == 0, "network_guard_triggered")
        require(len(self.requests) == 2 and len(self.search_results) == 1, "incomplete_agent_loop")
        first, second = self.requests
        require(first["messages"] == initial, "invalid_initial_history")
        require(second["messages"][:-2] == first["messages"], "incomplete_followup_history")
        require(second["messages"][-2] == {"role": "assistant", "content": SEARCH_TEXT}, "missing_tool_call")
        observation = second["messages"][-1]
        require(observation["role"] == "tool" and isinstance(observation["content"], str), "invalid_tool_observation")
        body = observation["content"]
        if failure:
            require(self.search_results == ["Error"] and body.startswith("Error:"), "missing_error_observation")
        else:
            result = self.search_results[0]
            require(isinstance(result, dict) and len(result["data"]) == 5, "invalid_mock_results")
            require("found 5 results" in body and "[mock]" in body and QUERY in body, "missing_mock_observation")
        require(len(self.search_requests) == (3 if failure else 0), "invalid_search_request_count")
        require(len(self.search_clients) == (1 if failure else 0), "invalid_search_client_count")
        require(all(response.is_closed for response in self.search_responses), "search_response_not_closed")
        require(all(client.is_closed for client in self.search_clients), "search_client_not_closed")
        require(
            len(self.model_clients) == 1 and all(client.is_closed() for client in self.model_clients),
            "model_client_not_closed",
        )
        require(len(self.closed_environments) == 1, "environment_not_closed")
        require(self.events == ["model", "search", "model", "close"], "invalid_agent_event_order")
        metadata = output["metadata"]
        expected_error = "search_failed" if failure else None
        require(
            metadata["stop_reason"] == "env_done" and metadata["final_answer"] == self.answer, "missing_final_answer"
        )
        require(metadata["last_error"] == expected_error, "invalid_last_error")
        require(
            metadata["branch_counts"] == {"answer": 1, "code": 0, "tool_call": 1, "format_error": 0},
            "invalid_branch_counts",
        )
        return {
            "scenario": self.scenario,
            "backend": "mock" if not failure else self.scenario.removesuffix("-error"),
            "model_requests": len(self.requests),
            "search_calls": len(self.search_results),
            "search_http_requests": len(self.search_requests),
            "network_attempts": self.network_attempts,
            "environment_closed": True,
            "model_clients_closed": True,
            "search_clients_closed": True,
            "final_answer": metadata["final_answer"],
            "stop_reason": metadata["stop_reason"],
            "last_error": metadata["last_error"],
            "observation": body,
            "events": self.events,
        }


def run_smoke(scenario: str, directory: Path) -> dict[str, Any]:
    """在禁止网络与 sandbox 访问的环境执行 agent.main，返回并保存验证证据.

    scenario 支持 mock、retriever-error 和 external-error；directory
    自动创建，同名证据文件会被覆盖. 模型使用内存 HTTP transport，补丁在退出时恢复；验证条件不满足时抛出 RuntimeError.
    """

    require(scenario in SCENARIOS, "invalid_smoke_scenario")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    initial = [
        {"role": "system", "content": UNIFIED_SYSTEM_PROMPT},
        {"role": "user", "content": "请调用搜索工具查询 DeepEyes-V2 离线搜索验证，然后完成回答。"},
    ]
    input_path, output_path = directory / "input.json", directory / "output.json"
    write_json(
        input_path, {"messages": initial, "metadata": {"data_source": "offline-smoke", "data_index": "offline"}}
    )
    smoke = OfflineSmoke(scenario)
    executor = Mock(spec=agent.SandboxExecutor)
    executor.acquire_session.side_effect = RuntimeError("offline_sandbox_access_forbidden")
    real_close = env_deepeyes_v2.DeepEyesV2Env.close

    async def close_environment(environment: env_deepeyes_v2.DeepEyesV2Env) -> None:
        await real_close(environment)
        smoke.closed_environments.append(environment)
        smoke.events.append("close")

    with ExitStack() as patches:
        patches.enter_context(patch.dict(os.environ, smoke_environment(scenario, directory), clear=True))
        require(load_search_config().backend == scenario.removesuffix("-error"), "unexpected_backend")
        for target, name, replacement in (
            (socket.socket, "connect", smoke.deny_network),
            (socket.socket, "connect_ex", smoke.deny_network),
            (socket, "create_connection", smoke.deny_network),
            (socket, "getaddrinfo", smoke.deny_network),
            (httpx.HTTPTransport, "handle_request", smoke.deny_network),
            (httpx.AsyncHTTPTransport, "handle_async_request", smoke.deny_async_network),
            (openai, "AsyncOpenAI", smoke.model_client),
            (search_http, "_create_client", smoke.search_client),
            (env_deepeyes_v2, "search", smoke.search),
            (env_deepeyes_v2.DeepEyesV2Env, "close", close_environment),
        ):
            patches.enter_context(patch.object(target, name, replacement))
        patches.enter_context(patch.object(agent, "_build_executor", return_value=executor))
        patches.enter_context(
            patch.object(
                sys, "argv", ["app.agent", "--input-json", str(input_path), "--output-json", str(output_path)]
            )
        )
        try:
            agent.main()
        finally:
            # 运行器负责关闭注入的模型客户端；环境清理由真实 agent 执行。
            asyncio.run(smoke.close_model_clients())
        executor.acquire_session.assert_not_called()
    output = json.loads(output_path.read_text(encoding="utf-8"))
    report = smoke.verify(initial, output)
    write_json(directory / "model_requests.json", smoke.requests)
    write_json(directory / "report.json", report)
    return report


def main() -> None:
    """解析离线 smoke 参数，执行指定场景并输出 JSON 验证报告."""

    parser = argparse.ArgumentParser(description="离线执行真实 agent 搜索循环并验证最终答案。")
    parser.add_argument(
        "--scenario", choices=SCENARIOS, default="mock", help="默认使用 mock；错误场景提供受控 503 响应。"
    )
    parser.add_argument("--output-dir", type=Path, help="保存输入、输出、模型请求和验证报告的目录。")
    args = parser.parse_args()
    directory = args.output_dir or EXAMPLE_DIR / "log" / "search-offline" / args.scenario
    report = run_smoke(args.scenario, directory)
    sys.stdout.write(json.dumps(report, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
