import os

# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        # model="google/gemma-4-E2B",
        model="google/gemma-4-31B",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )
    print("Starting generation")
    outputs = llm.generate(
        ["Explain quantum computing in simple terms."],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    print("Generation complete")
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
