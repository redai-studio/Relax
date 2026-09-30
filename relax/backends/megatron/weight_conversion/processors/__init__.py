from .padding_remover import remove_padding


try:
    from .quantizer_compressed_tensors import quantize_params_compressed_tensors
    from .quantizer_fp8 import quantize_params_fp8
except Exception:
    quantize_params_fp8 = None
    quantize_params_compressed_tensors = None


__all__ = ["remove_padding", "quantize_params", "quantize_params_fp8", "quantize_params_compressed_tensors"]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    # Kimi K3-style mxfp4-pack-quantized releases: routed experts are already
    # packed by Bridge's quantized mappings inside BridgeConverter, and every
    # other namespace is ignored per config. The INT4 packer below must not
    # touch the emitted _packed/_scale pairs (or the BF16 router gate).
    if quantization_config.get("format") == "mxfp4-pack-quantized":
        return converted_named_params
    quant_method = quantization_config["quant_method"]
    if quant_method == "fp8":
        if quantize_params_fp8 is None:
            raise NotImplementedError("fp8 quantization is not supported in this environment.")
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quant_method == "compressed-tensors":
        if quantize_params_compressed_tensors is None:
            raise NotImplementedError("compressed-tensors quantization is not supported in this environment.")
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
    raise ValueError(f"Unsupported quantization method: {quant_method!r}")
