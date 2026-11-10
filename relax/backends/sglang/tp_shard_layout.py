# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Read-only native-loader qualification of ordinary TP rectangles.

The probe executes on a detached module graph containing meta tensors only.
Canonical inputs, native target aliases and source dependency groups determine
one indivisible route. Unsupported components remain in the native exporter.
"""

import copy
import functools
import types
from collections import defaultdict
from enum import Enum
from math import prod

import torch
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from relax.distributed.checkpoint_service.backends.shard_p2p.layout import TensorRegion
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint, region_write, validate_coverage


class _ProbeRuntime:
    """Detached coordinates only; runtime operations cannot touch the live
    worker."""

    def __init__(self, kind: str, coordinates: tuple = ()) -> None:
        self.kind, self.coordinates = kind, coordinates

    def __getattr__(self, name):
        for key, value in object.__getattribute__(self, "coordinates"):
            if key == name:
                return value
        raise NotImplementedError("Native loader accesses a runtime handle during meta qualification")

    def _reject(self, *args, **kwargs):
        raise NotImplementedError("Native loader operates on a runtime handle during meta qualification")

    __call__ = __bool__ = __iter__ = __len__ = __getitem__ = __enter__ = __exit__ = _reject
    __eq__ = __ne__ = __hash__ = _reject


def _runtime_metadata(value):
    """Recognize SGLang's group coordinates and PyTorch runtime handles."""
    cls = type(value)
    if cls.__module__ == "sglang.srt.distributed.parallel_state" and cls.__name__ == "GroupCoordinator":
        fields = ("rank", "local_rank", "world_size", "rank_in_group", "is_first_rank", "is_last_rank")
        values = []
        for field in fields:
            item = getattr(value, field)
            if type(item) not in (int, bool):
                raise NotImplementedError("Unrecognized SGLang group coordinate")
            values.append((field, item))
        return _ProbeRuntime("sglang-group", tuple(values))
    handles = (
        torch.distributed.ProcessGroup,
        torch.cuda.Stream,
        torch.cuda.Event,
        getattr(torch._C._distributed_c10d, "_SymmetricMemory", None),
    )
    if isinstance(value, tuple(item for item in handles if isinstance(item, type))):
        return _ProbeRuntime(cls.__module__ + "." + cls.__name__)
    return None


def _registered_operator(value) -> bool:
    if isinstance(value, torch._ops.OpOverloadPacket):
        namespace, name = value._qualified_op_name.split("::", 1)
        return getattr(getattr(torch.ops, namespace), name) is value
    if isinstance(value, torch._ops.OpOverload):
        namespace, name = value._schema.name.split("::", 1)
        packet = getattr(getattr(torch.ops, namespace), name)
        return getattr(packet, value._schema.overload_name or "default") is value
    return False


def _is_lazy(value) -> bool:
    cls = type(value)
    return cls.__module__ == "sglang.srt.utils.common" and cls.__name__ == "LazyValue"


def _leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)


def _meta_model(model: torch.nn.Module) -> torch.nn.Module:
    """Clone structure and aliasing without reading or copying CUDA bytes."""
    memo, storages = {}, {}
    module_dicts = {id(module.__dict__) for module in model.modules()}
    for module in model.modules():
        memo[id(module)] = object.__new__(type(module))

    def member(key, value):
        try:
            return replace(value)
        except NotImplementedError as error:
            label = key if isinstance(key, str) else type(key).__name__
            raise NotImplementedError(f"{label}: {error}") from error

    def replace(value):
        if id(value) in memo:
            return memo[id(value)]
        runtime = _runtime_metadata(value)
        if runtime is not None:
            memo[id(value)] = runtime
            return runtime
        if _registered_operator(value):
            return value  # Registered code, not model-owned state or a raw capsule.
        if isinstance(value, torch.Tensor):
            key = StorageWeakRef(value.untyped_storage())
            if key not in storages:
                storages[key] = torch.empty(value.untyped_storage().nbytes(), dtype=torch.uint8, device="meta")
            capacity = storages[key].numel() // value.element_size() * value.element_size()
            view = (
                storages[key][:capacity]
                .view(value.dtype)
                .as_strided(value.shape, value.stride(), value.storage_offset())
            )
            result = torch.Tensor._make_subclass(type(value), view, value.requires_grad)
            memo[id(value)] = result
            result.__dict__.update({key: replace(item) for key, item in value.__dict__.items()})
            return result
        if isinstance(value, types.MethodType):
            return types.MethodType(replace(value.__func__), replace(value.__self__))
        if isinstance(value, types.BuiltinFunctionType):
            owner = value.__self__
            if owner is None or isinstance(owner, (types.ModuleType, type)):
                return value  # Immutable code, without a live model instance.
            bound = replace(owner)
            result = (
                _ProbeRuntime("bound-runtime:" + value.__name__)
                if isinstance(bound, _ProbeRuntime)
                else getattr(bound, value.__name__)
            )
            memo[id(value)] = result
            return result
        if isinstance(value, types.FunctionType):
            cells = tuple(types.CellType(None) for _ in (value.__closure__ or ()))
            result = types.FunctionType(value.__code__, value.__globals__, value.__name__, closure=cells or None)
            memo[id(value)] = result
            for source, target in zip(value.__closure__ or (), cells, strict=True):
                target.cell_contents = replace(source.cell_contents)
            result.__defaults__ = replace(value.__defaults__)
            result.__kwdefaults__ = replace(value.__kwdefaults__)
            result.__dict__.update(replace(value.__dict__))
            return result
        if isinstance(value, functools.partial):
            return functools.partial(replace(value.func), *replace(value.args), **replace(value.keywords))
        if isinstance(value, dict):
            result = copy.copy(value)
            result.clear()
            memo[id(value)] = result
            result.update(
                {
                    replace(key): member(key, item)
                    for key, item in value.items()
                    if id(value) not in module_dicts or key not in protocol_state
                }
            )
            return result
        if isinstance(value, list):
            result = []
            memo[id(value)] = result
            result.extend(replace(item) for item in value)
            return result
        if isinstance(value, tuple):
            return tuple(replace(item) for item in value)
        if isinstance(value, set):
            return {replace(item) for item in value}
        if value is None or isinstance(
            value, (str, bytes, int, float, bool, complex, Enum, type, torch.dtype, torch.device, torch.layout)
        ):
            return value
        if hasattr(value, "__dict__"):
            try:
                result = (
                    types.SimpleNamespace()
                    if isinstance(value, types.SimpleNamespace)
                    else object.__new__(type(value))
                )
            except TypeError as error:
                raise NotImplementedError("Native loader contains an opaque mutable object") from error
            memo[id(value)] = result
            result.__dict__.update(replace(value.__dict__))
            if any(getattr(cls, "__slots__", ()) for cls in type(value).__mro__):
                raise NotImplementedError("Native loader contains unqualified slotted state")
            return result
        raise NotImplementedError(f"Native loader contains opaque state: {type(value).__name__}")

    protocol_state = {
        "_relax_native_weight_guard",
        "_relax_native_weight_lease",
        "_relax_expert_ipc",
        "_relax_expert_delivery",
        "_relax_weight_sync_reply",
    }
    for module in model.modules():
        # Memoize the live dict identity. A temporary filtered dict can be freed
        # and its id reused for another module, corrupting the cloned graph.
        memo[id(module)].__dict__ = replace(module.__dict__)
        if callable(getattr(module, "load_weights", None)):
            replace(module.load_weights)
    return memo[id(model)]


def _matrix_view(value: torch.Tensor) -> tuple[int, int, int]:
    if not value.ndim or not all(value.shape):
        raise NotImplementedError("Empty native-loader view")
    expected = 1
    for size, stride in reversed(list(zip(value.shape[1:], value.stride()[1:], strict=True))):
        if size != 1 and stride != expected:
            raise NotImplementedError("Native-loader view has a noncontiguous inner dimension")
        expected *= size
    return value.shape[0], prod(value.shape[1:]), value.stride(0)


def _lazy_factory_signature(value, modules: set[int]) -> tuple:
    state = vars(value)
    if set(state) != {"_creator", "_value"} or state["_value"] is not None:
        raise NotImplementedError("Native lazy cache is not an unresolved creator")
    creator = state["_creator"]
    if not isinstance(creator, types.FunctionType) or creator.__dict__:
        raise NotImplementedError("Native lazy cache creator is unsupported")

    def captured(item):
        if isinstance(item, torch.nn.Module) and id(item) in modules:
            return ("module", id(item))
        if item is None or isinstance(item, (str, bytes, int, float, bool, complex, Enum, type)):
            return (type(item), item)
        raise NotImplementedError("Native lazy cache captures mutable non-model state")

    return (
        creator.__code__,
        id(creator.__globals__),
        tuple(captured(cell.cell_contents) for cell in creator.__closure__ or ()),
        tuple(captured(item) for item in creator.__defaults__ or ()),
        tuple((key, captured(item)) for key, item in (creator.__kwdefaults__ or {}).items()),
    )


def _state_schema(model: torch.nn.Module) -> tuple:
    """Reject derived state, rebinding and Python mutations bypassed by direct
    loads."""
    seen, caches, tensors = set(), [], []
    checking_cache = False

    def walk(value):
        if _registered_operator(value):
            return ("operator", id(value))
        if _is_lazy(value):
            if id(value) in seen or checking_cache:
                raise NotImplementedError("Shared or nested native lazy caches require an explicit adapter")
            seen.add(id(value))
            if set(vars(value)) != {"_creator", "_value"}:
                raise NotImplementedError("Native lazy cache has an unsupported state layout")
            caches.append(value)
            return ("lazy-slot", len(caches) - 1)
        if isinstance(value, _ProbeRuntime):
            return (
                "runtime",
                id(value),
                object.__getattribute__(value, "kind"),
                object.__getattribute__(value, "coordinates"),
            )
        if id(value) in seen:
            return ("alias", id(value))
        if isinstance(value, torch.Tensor):
            seen.add(id(value))
            tensors.append(value)
            return (
                "tensor",
                id(value),
                StorageWeakRef(value.untyped_storage()),
                tuple(value.shape),
                tuple(value.stride()),
                value.storage_offset(),
                value.dtype,
                walk(value.__dict__),
            )
        if value is None or isinstance(
            value, (str, bytes, int, float, bool, complex, Enum, type, torch.dtype, torch.device, torch.layout)
        ):
            return (type(value), value)
        seen.add(id(value))
        if isinstance(value, dict):
            return ("dict", id(value), tuple((walk(key), walk(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return (type(value), id(value), tuple(walk(item) for item in value))
        if isinstance(value, set):
            return ("set", id(value), frozenset(walk(item) for item in value))
        if isinstance(value, types.MethodType):
            return ("method", id(value), walk(value.__func__), walk(value.__self__))
        if isinstance(value, types.BuiltinFunctionType):
            owner = value.__self__
            return (
                "builtin",
                id(value),
                value.__name__,
                None if owner is None or isinstance(owner, (types.ModuleType, type)) else walk(owner),
            )
        if isinstance(value, types.FunctionType):
            return (
                "function",
                id(value),
                tuple(walk(cell.cell_contents) for cell in value.__closure__ or ()),
                walk(value.__defaults__),
                walk(value.__kwdefaults__),
                walk(value.__dict__),
            )
        if isinstance(value, functools.partial):
            return ("partial", id(value), walk(value.func), walk(value.args), walk(value.keywords))
        return (type(value), id(value), walk(value.__dict__))

    structure = walk(model)
    main_seen = seen.copy()
    checking_cache = True
    snapshots = []
    for cache in caches:
        seen = main_seen.copy()
        snapshots.append((cache, walk(vars(cache))))
    return structure, snapshots, tensors


def _qualify_lazy_refresh(model, before: tuple, after: tuple) -> bool:
    structure, previous, _ = before
    current_structure, current, _ = after
    if current_structure != structure or len(previous) != len(current):
        raise NotImplementedError("Native loader changes derived or structural model state")
    modules = {id(module) for module in model.modules()}
    changed = {}

    def immutable(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool, complex, Enum, type)):
            return True
        return isinstance(value, (tuple, frozenset)) and all(immutable(item) for item in value)

    for index, ((old, old_state), (new, new_state)) in enumerate(zip(previous, current, strict=True)):
        if new is old:
            if new_state != old_state:
                raise NotImplementedError("Native loader mutates an existing lazy cache")
            if vars(new)["_creator"] is None:
                if not immutable(vars(new)["_value"]):
                    raise NotImplementedError("Unrefreshed native lazy cache retains mutable state")
            else:
                _lazy_factory_signature(new, modules)
        else:
            changed[index] = _lazy_factory_signature(new, modules)
    if not changed:
        return False

    # Exercise the warm state too: an empty reload must invalidate a resolved
    # weight cache, while unrelated runtime caches remain unchanged.
    with _CopyProbe(model, [], forbid_writes=True):
        for index in changed:
            current[index][0].value
    with _CopyProbe(model, [], forbid_writes=True):
        model.load_weights([])
    empty_structure, refreshed, _ = _state_schema(model)
    if empty_structure != structure or len(refreshed) != len(current):
        raise NotImplementedError("Empty native refresh changes model storage or cache semantics")
    for index, ((old, old_state), (new, new_state)) in enumerate(zip(current, refreshed, strict=True)):
        if index in changed:
            if new is old or _lazy_factory_signature(new, modules) != changed[index]:
                raise NotImplementedError("Empty native refresh does not invalidate every lazy cache")
        elif new is not old or new_state != old_state:
            raise NotImplementedError("Empty native refresh changes an unrelated lazy cache")
    return True


class _CopyProbe(TorchDispatchMode):
    def __init__(self, model, inputs, *, forbid_writes: bool = False, persistent_tensors=()) -> None:
        self.forbid_writes = forbid_writes
        self.origins = {
            StorageWeakRef(value.untyped_storage()): (name, tuple(value.shape), str(value.dtype))
            for name, value in inputs
        }
        self.provenance = {key: {value[0]} for key, value in self.origins.items()}
        self.targets, self.parameters, self.edges = {}, dict(model.named_parameters(remove_duplicate=False)), []
        self.copies, self.bad_hf, self.bad_targets = [], set(), set()
        self.unattributed_writes = set()
        self.native_edges = []
        aliases = defaultdict(list)
        for name, value in self.parameters.items():
            aliases[StorageWeakRef(value.untyped_storage())].append(name)
            if not value.is_contiguous() or value.dtype != torch.bfloat16:
                self.bad_targets.add(name)
        for storage, names in aliases.items():
            self.targets[storage] = (names[0], self.parameters[names[0]].storage_offset())
            if len(names) > 1:
                self.bad_targets.update(names)
        for name, value in model.named_buffers(remove_duplicate=False):
            storage = StorageWeakRef(value.untyped_storage())
            if storage in self.targets:
                self.bad_targets.add(self.targets[storage][0])
            else:
                self.targets[storage] = ("buffer:" + name, value.storage_offset())
                self.bad_targets.add("buffer:" + name)
        for index, value in enumerate(persistent_tensors):
            storage = StorageWeakRef(value.untyped_storage())
            if storage not in self.targets:
                name = f"state:{index}"
                self.targets[storage] = (name, value.storage_offset())
                self.bad_targets.add(name)
        self.native_provenance = {storage: {name} for storage, (name, _) in self.targets.items()}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = list(_leaves((args, kwargs)))
        if any(value.device.type == "cuda" for value in tensors):
            raise RuntimeError("TP probe attempted to access a live CUDA tensor")
        if kwargs.get("device") is not None and torch.device(kwargs["device"]).type == "cuda":
            raise NotImplementedError("Native loader explicitly allocates CUDA storage during probing")
        if func is torch.ops.aten._local_scalar_dense.default:
            raise NotImplementedError("Native loader has a data-dependent branch")
        origins = set().union(*(self.provenance.get(StorageWeakRef(v.untyped_storage()), set()) for v in tensors))
        native_origins = set().union(
            *(self.native_provenance.get(StorageWeakRef(v.untyped_storage()), set()) for v in tensors)
        )
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            if self.forbid_writes:
                raise NotImplementedError("Empty native refresh mutates tensor storage")
            destination = args[index] if index < len(args) else kwargs.get(argument.name)
            if not isinstance(destination, torch.Tensor) or destination.device.type != "meta":
                raise RuntimeError("TP probe attempted to mutate live or unsupported storage")
            storage = StorageWeakRef(destination.untyped_storage())
            if storage in self.origins:
                self.bad_hf.add(self.origins[storage][0])
            target = self.targets.get(storage)
            if target is None:
                continue  # Meta temporaries are traced through provenance below.
            name, base = target
            if not origins:
                self.unattributed_writes.add(name)
            self.edges.extend((hf, name) for hf in origins)
            self.native_edges.extend((origin, name) for origin in native_origins)
            try:
                if func is not torch.ops.aten.copy_.default or len(args) < 2:
                    raise NotImplementedError("Native loader transforms the destination")
                source = args[1]
                origin = self.origins.get(StorageWeakRef(source.untyped_storage()))
                if origin is None or source.dtype != destination.dtype or source.shape != destination.shape:
                    raise NotImplementedError("Native loader changes source values, dtype or geometry")
                hf, shape, dtype = origin
                rows, columns, stride = _matrix_view(source)
                _, _, destination_stride = _matrix_view(destination)
                width = prod(shape[1:])
                row, column = divmod(source.storage_offset(), width)
                if stride != width or column + columns > width or row + rows > shape[0]:
                    raise NotImplementedError("Native copy is not one canonical rectangle")
                self.copies.append(
                    dict(
                        hf=hf,
                        param=name,
                        global_shape=[shape[0], width],
                        offset=[row, column],
                        shape=[rows, columns],
                        storage_offset=destination.storage_offset() - base,
                        stride=destination_stride,
                        dtype=dtype,
                    )
                )
            except NotImplementedError:
                self.bad_targets.add(name)
        result = func(*args, **kwargs)
        for value in _leaves(result):
            self.provenance.setdefault(StorageWeakRef(value.untyped_storage()), set()).update(origins)
            self.native_provenance.setdefault(StorageWeakRef(value.untyped_storage()), set()).update(native_origins)
        return result


def _writes(records, targets, tp_rank: int) -> list[dict]:
    sources = defaultdict(list)
    for record in records:
        for source in record["regions"]:
            sources[source["hf"]].append((record["rank"], source))
    result = []
    for target in targets:
        for rank, source in sources[target["hf"]]:
            if source["global_shape"] != target["global_shape"] or source["dtype"] != target["dtype"]:
                raise ValueError("Canonical TP source and native target disagree")
            write = region_write(
                TensorRegion.from_descriptor(source, itemsize=2),
                TensorRegion.from_descriptor(target, itemsize=2),
                id=f"ordinary:{tp_rank}:{len(result)}",
                source=source["source"],
                source_rank=rank,
                param=target["param"],
                target=f"tp-{tp_rank}",
                hf=target["hf"],
            )
            if write is not None:
                result.append(write)
    return result


@torch.no_grad()
def describe_tp_targets(model: torch.nn.Module, description: dict, tp_rank: int) -> dict:
    """Qualify complete dependency components against the actual loader."""
    records, specs, world = description["records"], description["native_specs"], description["source_world_size"]
    candidate_hf = {region["hf"] for record in records for region in record["regions"]}
    clone = _meta_model(model)
    dtypes = {str(dtype): dtype for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int64, torch.int32)}
    inputs = [
        (name, torch.empty(spec["shape"], dtype=dtypes[spec["dtype"]], device="meta"))
        for name, spec in sorted(specs.items())
    ]
    state_before = _state_schema(clone)
    recorder = _CopyProbe(clone, inputs, persistent_tensors=state_before[2])

    def signature(value):
        return (
            StorageWeakRef(value.untyped_storage()),
            value.shape,
            value.stride(),
            value.storage_offset(),
            value.dtype,
        )

    before = {name: signature(value) for name, value in recorder.parameters.items()}
    with recorder:
        clone.load_weights(inputs)
    native_refresh = _qualify_lazy_refresh(clone, state_before, _state_schema(clone))
    if dict(clone.named_parameters(remove_duplicate=False)).keys() != recorder.parameters.keys():
        raise NotImplementedError("Native loader changes the parameter inventory")
    if any(value is not recorder.parameters[name] for name, value in clone.named_parameters(remove_duplicate=False)):
        raise NotImplementedError("Native loader replaces its parameter objects")
    if any(signature(value) != before[name] for name, value in clone.named_parameters(remove_duplicate=False)):
        raise NotImplementedError("Native loader changes native storage layout")
    graph = defaultdict(set)

    def edge(left, right):
        graph[left].add(right)
        graph[right].add(left)

    for hf, target in recorder.edges:
        edge(("hf", hf), ("target", target))
    for source, target in recorder.native_edges:
        edge(("target", source), ("target", target))
    for record in records:
        for names in record["groups"].values():
            for name in names[1:]:
                edge(("hf", names[0]), ("hf", name))
    visited, selected, writes, layouts, components = set(), set(), [], {}, []
    for node in sorted(graph):
        if node in visited:
            continue
        pending, component = [node], set()
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component.add(item)
            pending.extend(graph[item] - component)
        visited.update(component)
        names = {name for kind, name in component if kind == "hf"}
        targets = {name for kind, name in component if kind == "target"}
        if not names and targets & recorder.unattributed_writes:
            raise NotImplementedError("Native loader mutates persistent storage without an HF dependency")
        if (
            not names
            or not targets
            or not names <= candidate_hf
            or names & recorder.bad_hf
            or targets & recorder.bad_targets
        ):
            continue
        shapes = {name: list(recorder.parameters[name].shape) for name in targets}
        copies = [copy for copy in recorder.copies if copy["param"] in targets]
        try:
            plan = _writes(records, copies, tp_rank)
            coverage = {name: dict(nbytes=prod(shape) * 2) for name, shape in shapes.items()}
            if validate_coverage(plan, coverage, source_world_size=world) != targets:
                continue
        except (ValueError, TypeError):
            continue  # An incomplete/overlapping component stays entirely native.
        selected.update(names)
        writes.extend(plan)
        layouts.update(shapes)
        components.append(sorted(names))
    # Write IDs are unique in the final union, independently of component order.
    for index, write in enumerate(writes):
        write["id"] = f"ordinary:{tp_rank}:{index}"
    return dict(
        description_hash=fingerprint(description),
        tp_rank=tp_rank,
        selected_hf=sorted(selected),
        components=components,
        target_shapes=layouts,
        target_hf={
            name: sorted({item["hf"] for item in recorder.copies if item["param"] == name}) for name in layouts
        },
        writes=writes,
        native_refresh=native_refresh,
    )
