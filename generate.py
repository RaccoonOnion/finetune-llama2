import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from IPython.core.debugger import set_trace

access_token = "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" # this token is used only for accessing lora weights and llama model, will be removed after finals

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    lora_weights: str = "RaccoonOnion/Llama-2-7b-hf-LoRA-alpaca-cleaned-rank8", # another option "RaccoonOnion/Llama-2-7b-hf-LoRA-alpaca-cleaned-rank1"
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, eos_token = '</s>')
    print("Currently running on:", device)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": device},token = access_token,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,token = access_token,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        print('loading')
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            is_trainable = True,
            torch_dtype=torch.float16,
        )
    print('done loading')

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.8,
        top_p=0.75,
        top_k=40,
        num_beams=4, # 4
        max_new_tokens=1024, # 128
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=3.0,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output).replace("</s>", "")
    
    # testing code
    for instruction in [
        # "Who is the president of United States?","Which city is the capital of China?", "1+1=?",
        # "Tell me about alpacas.",
        # "Tell me about the president of Mexico in 2019.",
        # "Tell me about the king of France in 2019.",
        # "List all Canadian provinces in alphabetical order.",
        # "Write a Python program that prints the first 10 Fibonacci numbers.",
        # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        # "Tell me five words that rhyme with 'shock'.",
        # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        # "Count up from 1 to 50.",
        # "What elements are in water?",
        # "Can you boil a metal bowl in a microwave?",
        # "Tell me the best way to cook scallop.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()

    with open('test.txt', 'r') as reader, open('test-output.txt', 'w') as writer:
        # Read and write each line, one at a time
        for line in reader:
            writer.write(evaluate(line)+"\n")
    print("Outputs are written to test-output.txt")

if __name__ == "__main__":
    fire.Fire(main)