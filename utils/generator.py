import os
import time
import math
import json
import torch
import requests

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Any, Tuple, Optional
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

CLOSED_WEIGHT_MODEL = {
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
    "deepseek-r1",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "qwen3.5-397b-a17b",
    "qwen-max-latest",
    "gpt-oss-120b"
}


def initial_model(path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.device_count() > 1:
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group("nccl", device_id=device)
        model = AutoModelForCausalLM.from_pretrained(path, tp_plan="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", attn_implementation="eager"
        )
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded from {path} on device {model.device}")
    return model, tokenizer


def batch_data(data_list: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


class GPTOpenAIGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("API_KEY", "")
        self.base_url = os.getenv("BASE_URL", "")

        self.timeout = 30.0
        print(f"Initialize with {self.model_name}")

        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating",
                "The server is overloaded or not ready yet",
                "upstream_error",
                "new_api_error",
                "Lock wait timeout exceeded",
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit",
                "429",
                "rate limit",
            ],
            "ignore_error": [
                "content",
                "reduce the length",
            ],
        }

    def _build_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _post_chat_completions(
        self,
        messages: List[Dict[str, str]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "API_KEY is not set. "
                "Please set environment variable API_KEY."
            )

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        payload.update(config)

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "python-requests/GPTOpenAIGenerator",
        }

        response = requests.post(
            self._build_url(),
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        # Include the server response body on non-2xx replies to aid debugging.
        if not response.ok:
            try:
                err_detail = response.json()
            except Exception:
                err_detail = response.text
            raise Exception(f"HTTP {response.status_code}: {err_detail}")

        try:
            return response.json()
        except Exception:
            raise Exception(f"Invalid JSON response: {response.text}")

    def generate_single(
        self, packed_data: Tuple[Dict[str, str], Dict[str, Any]]
    ) -> List[Tuple[str, float]]:
        item, config = packed_data

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": item["instruction"]},
        ]

        # Roughly preserve the chat context.
        if item.get("response"):
            messages.append({"role": "assistant", "content": item["response"]})

        while True:
            try:
                response = self._post_chat_completions(
                    messages=messages, config=config)
                results: List[Tuple[str, float]] = []
                for c in response.get("choices", []):
                    message = c.get("message", {}) or {}
                    content = (message.get("content") or "").strip()
                    results.append((content, 1.0))
                return results if results else [("", 0.0)]
            except Exception as e:
                err = str(e)
                print(err)

                continue_flag = False
                sleep_flag = False
                ignore_flag = False

                for x in self.error_types["continue_error"]:
                    if x in err:
                        continue_flag = True
                for x in self.error_types["sleep_error"]:
                    if x in err:
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types["ignore_error"]:
                    if x in err:
                        ignore_flag = True

                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)

                return [("", 0.0)]

    def generate(
        self,
        source: List[Dict[str, str]],
        config: Dict[str, Any],
    ) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        parallel = config.pop("parallel", False)
        config.pop("batch_size", None)

        print(source[0]["instruction"])
        packed_data = [(x, config) for x in source]

        if parallel:
            max_workers = max(1, os.cpu_count() // 2)
            with ProcessPoolExecutor(max_workers=max_workers):
                result: List[List[Tuple[str, float]]] = list(
                    process_map(
                        self.generate_single,
                        packed_data,
                        max_workers=max_workers,
                        chunksize=1,
                    )
                )
        else:
            result = [self.generate_single(x) for x in tqdm(packed_data)]

        return result


class VLLMGenerator(object):
    def __init__(
        self,
        model_name_or_path: str,
        lora_name_or_path: Optional[str] = None,
    ):
        def min_cuda_cc() -> float:
            """Return the minimum compute capability across all current CUDA devices."""
            if not torch.cuda.is_available():
                return 0.0
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return 0.0

            caps = []
            for i in range(device_count):
                p = torch.cuda.get_device_properties(i)
                caps.append(float(f"{p.major}.{p.minor}"))
            return min(caps)

        def infer_prequantized_method(model_ref: str) -> Optional[str]:
            """
            Best-effort detection of whether the model is already a pre-quantized checkpoint.
            Example return values: 'awq' / 'gptq' / 'bitsandbytes' / 'gguf' / 'fp8' / None
            """
            lower = model_ref.lower()
            p = Path(model_ref)

            # 1) Explicit GGUF paths or names.
            if lower.endswith(".gguf") or ".gguf:" in lower or lower.endswith("-gguf"):
                return "gguf"

            try:
                if p.is_file() and p.suffix.lower() == ".gguf":
                    return "gguf"
                if p.is_dir():
                    # A local directory containing GGUF files.
                    if any(x.suffix.lower() == ".gguf" for x in p.iterdir()):
                        return "gguf"

                    # Common AWQ marker file.
                    if (p / "quant_config.json").exists():
                        return "awq"

                    # Common GPTQ/AWQ marker file.
                    qcfg = p / "quantize_config.json"
                    if qcfg.exists():
                        try:
                            data = json.loads(qcfg.read_text(encoding="utf-8"))
                            qmethod = str(data.get("quant_method", "")).lower()
                            if "gptq" in qmethod:
                                return "gptq"
                            if "awq" in qmethod:
                                return "awq"
                        except Exception:
                            # The file exists but could not be parsed; keep probing.
                            pass
            except Exception:
                pass

            # 2) Try reading quantization_config from HF/local config.
            try:
                cfg = AutoConfig.from_pretrained(
                    model_ref,
                    trust_remote_code=True,
                )
                qconf = getattr(cfg, "quantization_config", None)

                if isinstance(qconf, dict):
                    qmethod = str(
                        qconf.get("quant_method")
                        or qconf.get("quantization_method")
                        or qconf.get("type")
                        or ""
                    ).lower()

                    if "awq" in qmethod:
                        return "awq"
                    if "gptq" in qmethod:
                        return "gptq"
                    if "bitsandbytes" in qmethod or "bnb" in qmethod:
                        return "bitsandbytes"
                    if "gguf" in qmethod:
                        return "gguf"
                    if "fp8" in qmethod:
                        return "fp8"
            except Exception:
                pass

            # 3) Weakest fallback: infer from the model name.
            if "awq" in lower:
                return "awq"
            if "gptq" in lower:
                return "gptq"
            if "bitsandbytes" in lower or "bnb" in lower:
                return "bitsandbytes"
            if "gguf" in lower:
                return "gguf"
            if "fp8" in lower:
                return "fp8"

            return None

        def detect_quantization(model_ref: str) -> Tuple[Optional[str], str]:
            """
            Choose a conservative default quantization strategy from model metadata
            and GPU capability.
            Returns: (quantization, dtype)
            """
            cc = min_cuda_cc()
            has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            preq = infer_prequantized_method(model_ref)

            # Without CUDA, do not proactively choose a GPU quantization mode.
            if not has_cuda:
                return None, "float32"

            # ---------- Respect pre-quantized checkpoints first ----------
            if preq == "gguf":
                # GGUF quantization is determined by the model file itself; keep dtype as auto.
                return "gguf", "auto"

            if preq == "awq":
                # AWQ: Turing (7.5)+; float16 is the safer dtype choice.
                if cc >= 7.5:
                    return "awq", "float16"
                return None, "float16"

            if preq == "gptq":
                # GPTQ: Volta (7.0)+
                if cc >= 7.0:
                    return "gptq", "float16"
                return None, "float16"

            if preq == "bitsandbytes":
                # Already pre-quantized or intended for the bitsandbytes loading path.
                return "bitsandbytes", "bfloat16" if cc >= 8.0 else "float16"

            if preq == "fp8":
                # For pre-quantized FP8 checkpoints, it is usually safer to let vLLM
                # infer quantization from the model config instead of forcing an override.
                return None, "bfloat16" if cc >= 8.9 else "float16"

            # ---------- Pick defaults for non-quantized models ----------
            # Ada/Hopper: prefer online FP8.
            if cc >= 8.9:
                return "fp8_per_tensor", "bfloat16"

            # Turing / Ampere: default to bitsandbytes 4-bit for broader compatibility.
            if cc >= 7.5:
                return "bitsandbytes", "bfloat16" if cc >= 8.0 else "float16"

            # Older GPUs: do not proactively quantize.
            return None, "float16"

        quantization, dtype = detect_quantization(model_name_or_path)

        self.quantization = quantization
        self.dtype = dtype

        llm_kwargs = dict(
            model=model_name_or_path,
            tensor_parallel_size=max(torch.cuda.device_count(), 1),
            dtype=dtype,
            trust_remote_code=True,
            max_model_len=8192,
            enable_lora=bool(lora_name_or_path),
        )
        # if quantization is not None:
        #     llm_kwargs["quantization"] = quantization

        self.llm = LLM(**llm_kwargs)

        self.tokenizer = self.llm.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.adapter = (
            LoRARequest(lora_name_or_path, 1, lora_name_or_path)
            if lora_name_or_path
            else None
        )

    def _build_prompts(self, source: List[Dict[str, str]]) -> List[str]:
        def build_chat_messages(item: Dict[str, str]) -> List[Dict[str, str]]:
            messages = [{"role": "user", "content": item["instruction"]}]
            response = item.get("response", "")
            if response:
                messages.append(
                    {"role": "assistant", "content": response + _MAGIC_SPLITTER_}
                )
            return messages

        def build_chat_prompt(tokenizer, item: Dict[str, str]) -> str:
            prompt = tokenizer.apply_chat_template(
                build_chat_messages(item),
                add_generation_prompt=True,
                tokenize=False,
            )
            if isinstance(prompt, list):
                prompt = tokenizer.decode(prompt)
            return prompt.split(_MAGIC_SPLITTER_)[0]

        return [build_chat_prompt(self.tokenizer, x) for x in source]

    def _filter_too_long_input(
        self, source: List[str], config: Dict[str, Any]
    ) -> List[str]:
        too_long_data_count = 0
        source_filtered = []

        for i, x in tqdm(
            enumerate(source),
            total=len(source),
            desc="filtering too long input",
        ):
            if len(self.tokenizer(x)["input_ids"]) > self.llm.llm_engine.model_config.max_model_len:
                source[i] = 'Output "TL;NR"'
                too_long_data_count += 1
            else:
                source_filtered.append(x)

        print(f"too long input count: {too_long_data_count}")
        if config["ignore_too_long"]:
            return source_filtered
        return source

    def generate(
        self,
        source: List[Dict[str, str]],
        config: Dict[str, Any],
    ) -> List[List[Tuple[str, float]]]:
        prompts = self._build_prompts(source)
        prompts = self._filter_too_long_input(prompts, config)
        print(prompts[0])

        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            n=config["n"],
            logprobs=1,
            stop=config["stop"]
            + (
                [self.tokenizer.eos_token]
                if getattr(self.tokenizer, "eos_token", None)
                else []
            ),
        )

        res_completions = []
        batch_instances = batch_data(prompts, batch_size=config["batch_size"])

        for prompt_batch in tqdm(batch_instances, total=len(batch_instances), desc="generating"):
            completions = self.llm.generate(
                prompt_batch, sampling_params, use_tqdm=False, lora_request=self.adapter)
            for output in completions:
                prompt_results = []
                for out in output.outputs:
                    total_logprob = 0
                    for x in out.logprobs:
                        token_idx = list(x.keys())[0]
                        if token_idx == self.pad_token_id or token_idx == self.eos_token_id:
                            break
                        total_logprob += x[token_idx].logprob
                    prompt_results.append((out.text, total_logprob))
                res_completions.append(prompt_results)

        return res_completions

    def likelihood(
        self,
        instances: List[Tuple[str, str]],
        config: Dict[str, Any],
    ) -> List[float]:
        def _batch_iter(xs, bs):
            for i in range(0, len(xs), bs):
                yield xs[i:i + bs]

        def _extract_logprob(lp_dict, token_id: int) -> float:
            if lp_dict is None:
                raise RuntimeError(
                    "Encountered a None entry in prompt_logprobs; cannot continue."
                )

            item = lp_dict.get(token_id, None)
            if item is None:
                raise RuntimeError(
                    f"token_id={token_id} was not found in prompt_logprobs; "
                    "this usually should not happen because vLLM returns the "
                    "logprob for the observed token."
                )

            if hasattr(item, "logprob"):
                return float(item.logprob)
            return float(item)

        batch_size = config["batch_size"]
        max_model_len = self.llm.llm_engine.model_config.max_model_len

        results = [0.0] * len(instances)
        vllm_inputs = []
        metas = []

        for idx, (input_text, output_text) in enumerate(instances):
            prompt_messages = [{"role": "user", "content": input_text}]
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = list(prompt_ids)

            full_messages = [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
            full_ids = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                continue_final_message=True,
            )
            full_ids = list(full_ids)

            if len(full_ids) > max_model_len:
                results[idx] = 0.0
                continue

            target_start = len(prompt_ids)
            target_end = len(full_ids)

            if target_start == target_end:
                results[idx] = 1.0
                continue

            vllm_inputs.append({"prompt_token_ids": full_ids})
            metas.append((idx, target_start, target_end))

        if not vllm_inputs:
            return results

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            n=1,
            logprobs=None,
            prompt_logprobs=1,
            detokenize=False,
        )

        for batch_inputs, batch_metas in zip(
            _batch_iter(vllm_inputs, batch_size),
            _batch_iter(metas, batch_size),
        ):
            outputs = self.llm.generate(
                batch_inputs,
                sampling_params,
                use_tqdm=False,
                lora_request=self.adapter,
            )

            for output, (orig_idx, target_start, target_end) in zip(outputs, batch_metas):
                prompt_token_ids = output.prompt_token_ids
                prompt_logprobs = output.prompt_logprobs

                if prompt_token_ids is None or prompt_logprobs is None:
                    raise RuntimeError(
                        "vLLM did not return prompt_token_ids or prompt_logprobs."
                    )

                if len(prompt_token_ids) != len(prompt_logprobs):
                    raise RuntimeError(
                        "prompt_token_ids and prompt_logprobs have different lengths."
                    )

                token_logprobs = []
                for pos in range(target_start, target_end):
                    token_id = prompt_token_ids[pos]
                    lp = _extract_logprob(prompt_logprobs[pos], token_id)
                    token_logprobs.append(lp)

                avg_logprob = (
                    sum(token_logprobs) /
                    len(token_logprobs) if token_logprobs else 0.0
                )
                results[orig_idx] = math.exp(avg_logprob)

        return results


def detect_generator(
    model_name_or_path: str,
    lora_name_or_path: Optional[str] = None,
) -> VLLMGenerator | GPTOpenAIGenerator:
    if model_name_or_path in CLOSED_WEIGHT_MODEL:
        return GPTOpenAIGenerator(model_name_or_path)
    return VLLMGenerator(model_name_or_path, lora_name_or_path)


def generate_with_llm(
    model_name_or_path: str,
    source: List[Dict[str, str]],
    config: Dict[str, Any],
    lora_name_or_path: Optional[str] = None,
) -> List[List[Tuple[str, float]]]:
    generator = detect_generator(model_name_or_path, lora_name_or_path)
    return generator.generate(source, config)


def likelihood_with_llm(
    model_name_or_path: str,
    instances: List[Tuple[str, str]],
    config: Dict[str, Any],
) -> List[float]:
    generator = detect_generator(model_name_or_path)
    if not hasattr(generator, "likelihood"):
        raise NotImplementedError(
            f"{generator.__class__.__name__} does not implement likelihood method."
        )
    return generator.likelihood(instances, config)


def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}

    for a, b, c in answers:
        x = str(b)
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c

    if not count:
        return "", ""

    return record[max(count, key=lambda x: count[x])]
