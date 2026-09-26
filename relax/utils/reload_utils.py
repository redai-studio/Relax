# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Module Reload Utilities.

Provides module hot-reload functionality, allowing Python modules to be
reloaded at runtime without restarting the service.
"""

import importlib
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class ReloadScope(Enum):
    """Define the reload scope of functions.

    Used to identify which component the function is loaded into, so RLSP can
    correctly distribute requests during hot-reloading.
    """

    ROLLOUT_MANAGER = "rollout_manager"  # Functions in RolloutManager
    DATA_SOURCE = "data_source"  # Functions in DataSource
    ACTOR = "actor"  # Functions in Actor
    IMMEDIATE = "immediate"  # Immediate loading, automatically reloaded on each call
    NOT_RELOADABLE = "not_reloadable"  # Not reloadable (bound during initialization)


@dataclass
class ReloadableFunction:
    """Description of a reloadable function.

    Attributes:
        name: Identifier name of the function, e.g., "rollout_function"
        attr_name: Attribute name on the object, e.g., "generate_rollout" (can be None for IMMEDIATE/NOT_RELOADABLE types)
        config_attr: Configuration attribute name in args, e.g., "rollout_function_path"
        scope: Reload scope, determines which component the function is loaded into
        required: Whether the function must be configured
        description: Description (used for API documentation)
    """

    name: str
    attr_name: Optional[str]
    config_attr: str
    scope: ReloadScope = ReloadScope.ROLLOUT_MANAGER
    required: bool = False
    description: str = ""


# Define the mapping of reloadable functions
# To add a new -path parameter, simply register it here to automatically support hot-reloading

RELOADABLE_FUNCTIONS: List[ReloadableFunction] = [
    # ========== RolloutManager Scope ==========
    # These functions are loaded and stored as members of RolloutManager
    ReloadableFunction(
        name="rollout_function",
        attr_name="generate_rollout",
        config_attr="rollout_function_path",
        scope=ReloadScope.ROLLOUT_MANAGER,
        required=True,
        description="Main rollout generation function",
    ),
    ReloadableFunction(
        name="eval_function",
        attr_name="eval_generate_rollout",
        config_attr="eval_function_path",
        scope=ReloadScope.ROLLOUT_MANAGER,
        required=True,
        description="Evaluation rollout function",
    ),
    ReloadableFunction(
        name="reward_post_process",
        attr_name="custom_reward_post_process_func",
        config_attr="custom_reward_post_process_path",
        scope=ReloadScope.ROLLOUT_MANAGER,
        required=False,
        description="Custom reward post-processing function",
    ),
    ReloadableFunction(
        name="convert_samples_to_train_data",
        attr_name="custom_convert_samples_to_train_data_func",
        config_attr="custom_convert_samples_to_train_data_path",
        scope=ReloadScope.ROLLOUT_MANAGER,
        required=False,
        description="Custom sample conversion function",
    ),
    # ========== DataSource Scope ==========
    # These functions are loaded and stored as members of DataSource
    ReloadableFunction(
        name="buffer_filter",
        attr_name="buffer_filter",
        config_attr="buffer_filter_path",
        scope=ReloadScope.DATA_SOURCE,
        required=False,
        description="Buffer filter function for RolloutDataSourceWithBuffer",
    ),
    # ========== Actor Scope ==========
    # These functions are loaded and stored as members of Actor
    ReloadableFunction(
        name="rollout_data_postprocess",
        attr_name="rollout_data_postprocess",
        config_attr="rollout_data_postprocess_path",
        scope=ReloadScope.ACTOR,
        required=False,
        description="Rollout data post-processing function in Actor",
    ),
    # ========== Immediate Scope ==========
    # Most functions are reloaded on each call; custom_rm is explicitly cached.
    ReloadableFunction(
        name="custom_loss_function",
        attr_name=None,
        config_attr="custom_loss_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom loss function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_tis_function",
        attr_name=None,
        config_attr="custom_tis_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom TIS function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_pg_loss_reducer_function",
        attr_name=None,
        config_attr="custom_pg_loss_reducer_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom PG loss reducer function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_generate_function",
        attr_name=None,
        config_attr="custom_generate_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom generate function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_rollout_log_function",
        attr_name=None,
        config_attr="custom_rollout_log_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom rollout log function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_eval_rollout_log_function",
        attr_name=None,
        config_attr="custom_eval_rollout_log_function_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom eval rollout log function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="dynamic_sampling_filter",
        attr_name=None,
        config_attr="dynamic_sampling_filter_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Dynamic sampling filter function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="custom_rm",
        attr_name=None,
        config_attr="custom_rm_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Custom reward model function (cached between explicit reloads)",
    ),
    ReloadableFunction(
        name="rollout_sample_filter",
        attr_name=None,
        config_attr="rollout_sample_filter_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Rollout sample filter function (auto-reloaded on each call)",
    ),
    ReloadableFunction(
        name="rollout_all_samples_process",
        attr_name=None,
        config_attr="rollout_all_samples_process_path",
        scope=ReloadScope.IMMEDIATE,
        required=False,
        description="Rollout all samples process function (auto-reloaded on each call)",
    ),
    # ========== Not Reloadable ==========
    # These functions are loaded only at initialization and do not support hot-reloading
    ReloadableFunction(
        name="data_source",
        attr_name=None,
        config_attr="data_source_path",
        scope=ReloadScope.NOT_RELOADABLE,
        required=False,
        description="DataSource class (only loaded at init)",
    ),
    ReloadableFunction(
        name="custom_model_provider",
        attr_name=None,
        config_attr="custom_model_provider_path",
        scope=ReloadScope.NOT_RELOADABLE,
        required=False,
        description="Custom model provider (only loaded at init)",
    ),
    ReloadableFunction(
        name="custom_megatron_init",
        attr_name=None,
        config_attr="custom_megatron_init_path",
        scope=ReloadScope.NOT_RELOADABLE,
        required=False,
        description="Custom Megatron initialization (only loaded at init)",
    ),
    ReloadableFunction(
        name="slime_router_middleware",
        attr_name=None,
        config_attr="slime_router_middleware_paths",
        scope=ReloadScope.NOT_RELOADABLE,
        required=False,
        description="Router middleware paths (only loaded at init)",
    ),
]


def get_reloadable_by_name(name: str) -> Optional[ReloadableFunction]:
    """Get the definition of a reloadable function by name."""
    for func_def in RELOADABLE_FUNCTIONS:
        if func_def.name == name:
            return func_def
    return None


def get_reloadable_by_scope(scope: ReloadScope) -> List[ReloadableFunction]:
    """Get a list of reloadable functions by scope."""
    return [f for f in RELOADABLE_FUNCTIONS if f.scope == scope]


def reload_function(module_path: str) -> Optional[Callable]:
    """Reload the function at the specified path.

    Args:
        module_path: Full path of the function, e.g., "module.submodule.function"

    Returns:
        Reloaded function object, or None if failed
    """
    if not module_path:
        return None

    try:
        py_module_path, _, attr_name = module_path.rpartition(".")
        if not py_module_path or not attr_name:
            logger.error(f"Invalid module path: {module_path}")
            return None

        # Reload Python module
        if py_module_path in sys.modules:
            py_module = sys.modules[py_module_path]
            importlib.reload(py_module)
            logger.info(f"Reloaded Python module: {py_module_path}")
        else:
            py_module = importlib.import_module(py_module_path)

        return getattr(py_module, attr_name)
    except Exception as e:
        logger.error(f"Failed to reload function from {module_path}: {e}")
        return None


def get_function_path(args: Any, func_def: ReloadableFunction) -> Optional[str]:
    """Get the function path from the configuration object.

    Args:
        args: Configuration object (usually argparse.Namespace)
        func_def: Definition of the reloadable function

    Returns:
        Function path or None
    """
    return getattr(args, func_def.config_attr, None)


def get_all_reloadable_paths(args: Any) -> Dict[str, Optional[str]]:
    """Get the path mapping of all reloadable functions.

    Args:
        args: Configuration object

    Returns:
        Mapping of function names to paths
    """
    return {func_def.name: get_function_path(args, func_def) for func_def in RELOADABLE_FUNCTIONS}


class ReloadableMixin:
    """Mixin class for reloadable modules.

    Classes using this Mixin need to:
    1. Have an args attribute (configuration object)
    2. Have corresponding function attributes (e.g., generate_rollout, eval_generate_rollout, etc.)

    Example:
        class RolloutManager(ReloadableMixin):
            def __init__(self, args):
                self.args = args
                self.generate_rollout = load_function(args.rollout_function_path)
                # ...

        # Usage:
        manager.reload_function("rollout_function")
    """

    def reload_function_by_name(self, module_name: str) -> Dict[str, Any]:
        """Reload the corresponding function by module name.

        Supports two module types:
        - Modules with attr_name (ROLLOUT_MANAGER scope): Update object attributes after reloading
        - Modules without attr_name (IMMEDIATE scope): Only refresh sys.modules cache

        Args:
            module_name: Module identifier name (e.g., "rollout_function", "custom_rm")

        Returns:
            Dictionary containing the reload result
        """
        # Find the corresponding function definition
        func_def = None
        for fd in RELOADABLE_FUNCTIONS:
            if fd.name == module_name:
                func_def = fd
                break

        if func_def is None:
            supported = [fd.name for fd in RELOADABLE_FUNCTIONS]
            return {
                "success": False,
                "error": f"Unknown module type: {module_name}",
                "supported_types": supported,
            }

        # Check if not reloadable
        if func_def.scope == ReloadScope.NOT_RELOADABLE:
            return {
                "success": False,
                "module_name": module_name,
                "scope": func_def.scope.value,
                "error": f"Module '{module_name}' is not reloadable (loaded only at initialization)",
            }

        # Get the module path
        module_path = get_function_path(self.args, func_def)
        if not module_path:
            return {
                "success": False,
                "module_name": module_name,
                "error": f"Module '{module_name}' is not configured",
            }

        # Reload the function (refresh sys.modules cache)
        new_func = reload_function(module_path)
        if new_func is None:
            return {
                "success": False,
                "module_name": module_name,
                "error": f"Failed to reload function from {module_path}",
            }

        # If there is an attr_name, update the object attribute
        if func_def.attr_name:
            old_func = getattr(self, func_def.attr_name, None)  # noqa: F841
            setattr(self, func_def.attr_name, new_func)
            logger.info(f"Updated {func_def.attr_name} from {module_path}")
            message = f"Module '{module_name}' reloaded and attribute updated"
        else:
            # IMMEDIATE scope: refresh module state; custom_rm also refreshes executor and workers.
            if module_name == "custom_rm":
                from relax.engine.rewards import RewardExecutor

                try:
                    RewardExecutor.reload_custom_reward(module_path)
                except Exception as exc:
                    logger.error(f"Failed to reload custom reward workers for {module_path}: {exc}")
                    return {
                        "success": False,
                        "module_name": module_name,
                        "module_path": module_path,
                        "scope": func_def.scope.value,
                        "error": f"Failed to reload custom reward workers: {exc}",
                    }
            logger.info(f"Refreshed sys.modules cache for {module_path}")
            message = f"Module '{module_name}' reloaded - will take effect on next call"

        return {
            "success": True,
            "module_name": module_name,
            "module_path": module_path,
            "scope": func_def.scope.value,
            "message": message,
        }

    def reload_module(self, module_name: str, module_path: str = None) -> Dict[str, Any]:
        """Hot-reload the specified module and update function references.

        This is the public interface for reload_function_by_name, supporting:
        - ROLLOUT_MANAGER scope: Update object attributes after reloading
        - IMMEDIATE scope: Refresh sys.modules cache
        - NOT_RELOADABLE scope: Return an error

        Args:
            module_name: Module identifier name (rollout_function, eval_function, custom_rm, etc.)
            module_path: Module path (optional, not used, provided by the registry)

        Returns:
            Dictionary containing the reload result
        """
        return self.reload_function_by_name(module_name)

    def get_loaded_modules(self) -> Dict[str, Optional[str]]:
        """Get information about currently loaded modules.

        Returns:
            Dictionary containing paths of all loaded modules
        """
        return get_all_reloadable_paths(self.args)

    def reload_all_functions(self) -> Dict[str, Any]:
        """Reload all configured reloadable functions.

        Returns:
            Dictionary containing the reload results for all functions
        """
        results = []

        for func_def in RELOADABLE_FUNCTIONS:
            module_path = get_function_path(self.args, func_def)
            if module_path:  # Only reload configured modules
                result = self.reload_function_by_name(func_def.name)
                results.append(
                    {
                        "module_name": func_def.name,
                        "success": result.get("success", False),
                        "message": result.get("message", ""),
                        "error": result.get("error"),
                    }
                )

        reloaded_count = sum(1 for r in results if r.get("success"))
        failed_count = len(results) - reloaded_count

        return {
            "success": failed_count == 0,
            "reloaded_count": reloaded_count,
            "failed_count": failed_count,
            "results": results,
            "message": f"Reloaded {reloaded_count}/{len(results)} modules",
        }

    def get_loaded_functions_info(self) -> Dict[str, Optional[str]]:
        """Get information about currently loaded functions.

        Returns:
            Mapping of function names to paths
        """
        return get_all_reloadable_paths(self.args)
