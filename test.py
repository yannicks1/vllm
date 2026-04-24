import os

# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="google/gemma-4-E2B",
        gpu_memory_utilization=0.45,
        enforce_eager=True,
    )
    print("Starting generation")
    outputs = llm.generate(
        ["Write a short hello world program in Python."],
        SamplingParams(),
    )
    print("Generation complete")
    print('geits guet? ')
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
