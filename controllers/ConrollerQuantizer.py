from time import time

import torch
from awq import AutoAWQForCausalLM
from optimum.gptq import GPTQQuantizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import get_gpu_utilization


class ControllerQuantizer:
    @staticmethod
    def awq(model_name: str, quant_config=None):
        metrics = {
            "gpu_background_ram": get_gpu_utilization(),
            "gpu_after_loading": 0,
            "gpu_quantized": 0,
            "load_time": 0,
            "quant_time": 0,
        }

        _quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
        if quant_config:
            _quant_config.update(quant_config)

        time_start = time()
        model = AutoAWQForCausalLM.from_pretrained(model_name, cache_dir=f"cache/")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=f"cache/{model_name}")
        metrics["load_time"] = time() - time_start
        model.to("cuda")
        metrics["gpu_after_loading"] = get_gpu_utilization()

        time_start = time()
        model.quantize(tokenizer, quant_config=_quant_config)
        metrics["quant_time"] = time() - time_start

        model.to("cuda")
        metrics["gpu_quantized"] = get_gpu_utilization()
        # try:
        #     model.save_quantized(f"cache/quantized/{model_name}/awq")
        # except:
        #     pass
        return model, metrics

    @staticmethod
    def gptq(model_name: str, quant_config=None):
        metrics = {
            "gpu_background_ram": get_gpu_utilization(),
            "gpu_after_loading": 0,
            "gpu_quantized": 0,
            "load_time": 0,
            "quant_time": 0,
        }
        _quant_config = {
            "bits": 4,
            "dataset": 'c4',
            "block_name_to_quantize": 'model.layers',
            "model_seqlen": 2048,
        }
        if quant_config:
            _quant_config.update(quant_config)

        time_start = time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache/")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir="cache/").to("cuda")
        metrics["load_time"] = time() - time_start
        model.to("cuda")
        metrics["gpu_after_loading"] = get_gpu_utilization()
        time_start = time()
        quantizer = GPTQQuantizer(**_quant_config)
        quantized_model = quantizer.quantize_model(model, tokenizer)
        metrics["quant_time"] = time() - time_start

        metrics["gpu_quantized"] = get_gpu_utilization()

        # try:
        #     model.save_pretrained(f"cache/quantized/{model_name}/gptq/{_quant_config['bits']}/{_quant_config['model_seqlen']}/{_quant_config['q_group_size']}")
        # except:
        #     pass

        return quantized_model, metrics

    @staticmethod
    def bitsandbytes(
            model_name: str,
            quant_config=None,
    ):
        metrics = {
            "gpu_background_ram": get_gpu_utilization(),
            "gpu_after_loading": "N/A",
            "gpu_quantized": 0,
            "load_time": 0,
            "quant_time": 0,
        }
        _quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            **quant_config
        )

        time_start = time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir="cache/",
            quantization_config=_quantization_config,
            device_map="auto"
        )
        metrics["load_time"] = time() - time_start
        metrics["quant_time"] = time() - time_start
        model.to("cuda")
        metrics["gpu_quantized"] = get_gpu_utilization()

        return model, metrics