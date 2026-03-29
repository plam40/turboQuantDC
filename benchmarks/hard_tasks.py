"""Hard Task Benchmarks — TurboQuant quality on real downstream tasks.

Runs Qwen2.5-3B-Instruct on GSM8K-style math, code generation, reasoning,
and factual recall under three conditions:
  1. FP16 baseline (normal inference)
  2. TQ-3 (3-bit TurboQuant KV cache)
  3. TQ-2 (2-bit aggressive compression)

Reports exact scores per category and an honest assessment of quality.

Usage:
    cd /home/dhawal/turboQuantDC && python benchmarks/hard_tasks.py
"""

from __future__ import annotations

import gc
import os
import re
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Allow running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512
CONDITIONS = ["FP16", "TQ-3", "TQ-2"]

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

MATH_TASKS = [
    {
        "prompt": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "18",
        "answer_variants": ["18", "$18", "18 dollars"],
    },
    {
        "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "3",
        "answer_variants": ["3", "3 bolts"],
    },
    {
        "prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "70000",
        "answer_variants": ["70000", "70,000", "$70,000", "$70000", "70000 dollars"],
    },
    {
        "prompt": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": "540",
        "answer_variants": ["540", "540 meters"],
    },
    {
        "prompt": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. If each chicken eats 3 cups of feed per day, how many cups of feed does she need to give her chickens in the final meal of the day?",
        "answer": "20",
        "answer_variants": ["20", "20 cups"],
    },
    {
        "prompt": "Kylar went to the store to get water and some apples. Apples cost $1.50 each and water costs $0.50 per bottle. Kylar bought 4 apples and 3 bottles of water. How much did everything cost?",
        "answer": "7.5",
        "answer_variants": ["7.5", "7.50", "$7.50", "$7.5", "7.50 dollars"],
    },
    {
        "prompt": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. If Seattle has 20 sheep, how many sheep do Toulouse and Charleston have together?",
        "answer": "240",
        "answer_variants": ["240", "240 sheep"],
    },
    {
        "prompt": "A merchant wants to make a choice of purchase between 2 purchasing plans: jewelry worth $5,000 or electronic gadgets worth $8,000. His financial advisor speculates that the jewelry market will go up 2.5% while the electronic gadgets market will rise 1.2% within the same month. If the merchant is looking for maximum profit, how much profit can the merchant get?",
        "answer": "125",
        "answer_variants": ["125", "$125", "125 dollars"],
    },
    {
        "prompt": "Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northward, covering 150 miles. What's the distance covered by each train in the two days?",
        "answer": "230",
        "answer_variants": ["230", "230 miles"],
    },
    {
        "prompt": "Jill gets paid $20 per hour to teach and $30 to be a personal trainer. If she works 50 hours a week and works 30 hours teaching, how much does she earn per week?",
        "answer": "1200",
        "answer_variants": ["1200", "1,200", "$1,200", "$1200"],
    },
    {
        "prompt": "Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she need to buy to make omelets for 4 weeks?",
        "answer": "7",
        "answer_variants": ["7", "7 dozen"],
    },
    {
        "prompt": "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then 2 hours to walk the next 8 miles. What was her average speed for the entire hike?",
        "answer": "4",
        "answer_variants": ["4", "4 miles per hour", "4 mph", "4 miles/hour"],
    },
    {
        "prompt": "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.50 each. It costs $3 a year to water and feed the tree. How many years will it take for the tree to pay for itself?",
        "answer": "13",
        "answer_variants": ["13", "13 years"],
    },
    {
        "prompt": "When Freda cooks canned tomatoes into sauce, they lose half their volume. Each can of tomatoes is 16 ounces. If she makes 32 ounces of sauce, how many cans does she need?",
        "answer": "4",
        "answer_variants": ["4", "4 cans"],
    },
    {
        "prompt": "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire class is enrolled in hip-hop dance?",
        "answer": "60",
        "answer_variants": ["60", "60%", "60 percent"],
    },
]

CODE_TASKS = [
    {
        "prompt": "Write a Python function called `is_palindrome` that returns True if a string is a palindrome (ignoring case and spaces). Only output the function, nothing else.",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal Panama') == True",
    },
    {
        "prompt": "Write a Python function called `fizzbuzz` that takes an integer n and returns a list of strings from 1 to n: 'FizzBuzz' for multiples of both 3 and 5, 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, and the number as a string otherwise. Only output the function.",
        "test": "result = fizzbuzz(15)\nassert result[0] == '1'\nassert result[2] == 'Fizz'\nassert result[4] == 'Buzz'\nassert result[14] == 'FizzBuzz'\nassert len(result) == 15",
    },
    {
        "prompt": "Write a Python function called `flatten` that takes a nested list and returns a flat list. For example, flatten([1, [2, [3, 4], 5]]) should return [1, 2, 3, 4, 5]. Only output the function.",
        "test": "assert flatten([1, [2, [3, 4], 5]]) == [1, 2, 3, 4, 5]\nassert flatten([]) == []\nassert flatten([1, 2, 3]) == [1, 2, 3]\nassert flatten([[1], [2], [3]]) == [1, 2, 3]",
    },
    {
        "prompt": "Write a Python function called `two_sum` that takes a list of integers and a target integer, and returns a tuple of two indices whose values add up to the target. Assume exactly one solution exists. Only output the function.",
        "test": "assert two_sum([2, 7, 11, 15], 9) in [(0, 1), (1, 0)]\nassert two_sum([3, 2, 4], 6) in [(1, 2), (2, 1)]",
    },
    {
        "prompt": "Write a Python function called `max_subarray_sum` that takes a list of integers and returns the maximum sum of any contiguous subarray (Kadane's algorithm). Only output the function.",
        "test": "assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\nassert max_subarray_sum([1]) == 1\nassert max_subarray_sum([-1, -2, -3]) == -1",
    },
    {
        "prompt": "Write a Python function called `is_valid_parens` that takes a string containing only '(', ')', '{', '}', '[', ']' and returns True if the brackets are balanced. Only output the function.",
        "test": "assert is_valid_parens('()[]{}') == True\nassert is_valid_parens('(]') == False\nassert is_valid_parens('([{}])') == True\nassert is_valid_parens('') == True",
    },
    {
        "prompt": "Write a Python function called `roman_to_int` that converts a Roman numeral string to an integer. Support I, V, X, L, C, D, M. Only output the function.",
        "test": "assert roman_to_int('III') == 3\nassert roman_to_int('IV') == 4\nassert roman_to_int('IX') == 9\nassert roman_to_int('MCMXCIV') == 1994",
    },
    {
        "prompt": "Write a Python function called `count_words` that takes a string and returns a dictionary mapping each word (lowercased) to its count. Only output the function.",
        "test": "result = count_words('the cat sat on the mat')\nassert result['the'] == 2\nassert result['cat'] == 1\nassert result['mat'] == 1",
    },
    {
        "prompt": "Write a Python function called `matrix_multiply` that takes two 2D lists (matrices) and returns their matrix product as a 2D list. Only output the function.",
        "test": "assert matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]\nassert matrix_multiply([[1]], [[2]]) == [[2]]",
    },
    {
        "prompt": "Write a Python function called `merge_sorted` that takes two sorted lists and returns a single sorted list. Do not use the built-in sort. Only output the function.",
        "test": "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\nassert merge_sorted([], [1, 2]) == [1, 2]\nassert merge_sorted([1], []) == [1]",
    },
]

REASONING_TASKS = [
    {
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer with only 'yes' or 'no'.",
        "answer": "no",
        "answer_variants": ["no"],
        "explanation": "Invalid syllogism - 'some flowers' doesn't necessarily include roses",
    },
    {
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Answer with just the number.",
        "answer": "5",
        "answer_variants": ["5", "5 cents", "$0.05", "0.05"],
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets? Answer with just the number.",
        "answer": "5",
        "answer_variants": ["5", "5 minutes"],
    },
    {
        "prompt": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake? Answer with just the number.",
        "answer": "47",
        "answer_variants": ["47", "47 days"],
    },
    {
        "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Answer with just the number.",
        "answer": "9",
        "answer_variants": ["9", "9 sheep"],
    },
    {
        "prompt": "If you overtake the person in second place in a race, what position are you in now? Answer with just the position.",
        "answer": "second",
        "answer_variants": ["second", "2nd", "2"],
    },
    {
        "prompt": "A doctor gives you 3 pills and tells you to take one every half hour. How many minutes will the pills last? Answer with just the number.",
        "answer": "60",
        "answer_variants": ["60", "60 minutes", "1 hour"],
    },
    {
        "prompt": "You have a 3-gallon jug and a 5-gallon jug. How do you measure exactly 4 gallons? Describe the steps briefly, then state the final answer: how many gallons are in the 5-gallon jug at the end?",
        "answer": "4",
        "answer_variants": ["4", "4 gallons"],
    },
    {
        "prompt": "Three people check into a hotel room that costs $30. They each pay $10. The manager realizes the room only costs $25, so he gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. Now each person paid $9 (total $27), the bellboy has $2. That's $29. Where is the missing dollar? Explain briefly.",
        "answer": "no missing dollar",
        "answer_variants": ["no missing", "there is no missing", "nowhere", "trick", "misleading", "fallacy", "accounting error"],
    },
    {
        "prompt": "Is the following statement true or false: 'This statement is false.' What type of logical problem is this? Answer briefly.",
        "answer": "paradox",
        "answer_variants": ["paradox", "liar paradox", "self-referential", "neither true nor false", "liar's paradox"],
    },
]

FACTUAL_TASKS = [
    {
        "prompt": "What is the capital of Australia? Answer with just the city name.",
        "answer": "Canberra",
        "answer_variants": ["Canberra"],
    },
    {
        "prompt": "What element has the chemical symbol 'Au'? Answer with just the element name.",
        "answer": "Gold",
        "answer_variants": ["Gold", "gold"],
    },
    {
        "prompt": "Who wrote the novel '1984'? Answer with just the author's name.",
        "answer": "George Orwell",
        "answer_variants": ["George Orwell", "Orwell", "Eric Arthur Blair"],
    },
    {
        "prompt": "What is the largest planet in our solar system? Answer with just the planet name.",
        "answer": "Jupiter",
        "answer_variants": ["Jupiter"],
    },
    {
        "prompt": "In what year did the Berlin Wall fall? Answer with just the year.",
        "answer": "1989",
        "answer_variants": ["1989"],
    },
    {
        "prompt": "What is the speed of light in vacuum, approximately, in km/s? Answer with just the number.",
        "answer": "300000",
        "answer_variants": ["300000", "300,000", "299792", "299,792", "3e5", "3 * 10^5", "approximately 300,000"],
    },
    {
        "prompt": "What programming language was created by Guido van Rossum? Answer with just the language name.",
        "answer": "Python",
        "answer_variants": ["Python"],
    },
    {
        "prompt": "What is the smallest prime number? Answer with just the number.",
        "answer": "2",
        "answer_variants": ["2"],
    },
    {
        "prompt": "What organ in the human body produces insulin? Answer with just the organ name.",
        "answer": "Pancreas",
        "answer_variants": ["Pancreas", "pancreas", "the pancreas"],
    },
    {
        "prompt": "How many chromosomes do humans typically have? Answer with just the number.",
        "answer": "46",
        "answer_variants": ["46", "23 pairs"],
    },
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result for a single task."""
    category: str
    task_idx: int
    prompt_short: str
    expected: str
    got: str
    correct: bool
    error: Optional[str] = None


@dataclass
class ConditionResults:
    """Results for one condition (FP16, TQ-3, TQ-2)."""
    condition: str
    results: List[TaskResult] = field(default_factory=list)

    def score(self, category: str) -> tuple[int, int]:
        cat_results = [r for r in self.results if r.category == category]
        correct = sum(1 for r in cat_results if r.correct)
        return correct, len(cat_results)

    def total_score(self) -> tuple[int, int]:
        correct = sum(1 for r in self.results if r.correct)
        return correct, len(self.results)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def extract_number(text: str) -> Optional[str]:
    """Extract the last number from text for math answers."""
    # Look for boxed answer first (common in chain-of-thought)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip().replace(',', '').replace('$', '')

    # Look for "the answer is X" pattern
    answer_match = re.search(
        r'(?:the answer is|answer:|final answer:?|therefore|thus|so)\s*\$?\s*([\d,]+\.?\d*)',
        text.lower()
    )
    if answer_match:
        return answer_match.group(1).replace(',', '')

    # Look for "= X" at the end
    eq_match = re.findall(r'=\s*\$?\s*([\d,]+\.?\d*)', text)
    if eq_match:
        return eq_match[-1].replace(',', '')

    # Fall back to last number in the text
    numbers = re.findall(r'(?<!\w)([\d,]+\.?\d*)(?!\w)', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def check_math_answer(response: str, task: dict) -> bool:
    """Check if the model's math response contains the correct answer."""
    # Check if any variant appears directly in the response
    response_lower = response.lower().strip()
    for variant in task["answer_variants"]:
        if variant.lower() in response_lower:
            return True

    # Try extracting the number
    extracted = extract_number(response)
    if extracted is not None:
        try:
            got = float(extracted)
            expected = float(task["answer"].replace(',', ''))
            return abs(got - expected) < 0.01
        except ValueError:
            pass

    return False


def check_code_answer(response: str, task: dict) -> tuple[bool, str]:
    """Extract function from response and run test cases."""
    # Strip markdown code blocks
    code = response
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', code, re.DOTALL)
    if code_blocks:
        code = code_blocks[0]
    else:
        # Try to extract just the function definition
        func_match = re.search(r'(def \w+\(.*?\n(?:(?:    |\t).*\n)*)', code)
        if func_match:
            code = func_match.group(1)

    # Clean up the code
    code = code.strip()

    # Execute the function definition + tests
    test_code = code + "\n\n" + task["test"]
    try:
        exec_globals: dict = {}
        exec(test_code, exec_globals)
        return True, ""
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_reasoning_answer(response: str, task: dict) -> bool:
    """Check if the reasoning response contains the correct answer."""
    response_lower = response.lower().strip()
    for variant in task["answer_variants"]:
        if variant.lower() in response_lower:
            return True
    return False


def check_factual_answer(response: str, task: dict) -> bool:
    """Check if the factual response contains the correct answer."""
    response_lower = response.lower().strip()
    for variant in task["answer_variants"]:
        if variant.lower() in response_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Chat formatting
# ---------------------------------------------------------------------------

def format_chat(prompt: str, tokenizer) -> str:
    """Format a prompt using the model's chat template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Give concise, direct answers."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def generate_fp16(
    model, tokenizer, prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS
) -> List[str]:
    """Generate responses using standard FP16 inference."""
    responses = []
    for prompt in prompts:
        chat_text = format_chat(prompt, tokenizer)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Decode only the generated tokens (strip the prompt)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response.strip())

        # Keep GPU memory tidy
        del output_ids, inputs
        torch.cuda.empty_cache()

    return responses


def generate_tq(
    model, tokenizer, prompts: List[str], bits: int,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> List[str]:
    """Generate responses using TurboQuant compressed KV cache."""
    from turboquantdc.hf_integration import TurboQuantCache

    responses = []
    for prompt in prompts:
        chat_text = format_chat(prompt, tokenizer)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        # Fresh TQ cache for each prompt
        tq_cache = TurboQuantCache(bits=bits)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                past_key_values=tq_cache,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response.strip())

        del output_ids, inputs, tq_cache
        torch.cuda.empty_cache()

    return responses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_math(responses: List[str], condition: str) -> List[TaskResult]:
    """Score math task responses."""
    results = []
    for i, (resp, task) in enumerate(zip(responses, MATH_TASKS)):
        correct = check_math_answer(resp, task)
        results.append(TaskResult(
            category="Math",
            task_idx=i,
            prompt_short=task["prompt"][:60] + "...",
            expected=task["answer"],
            got=resp[:200],
            correct=correct,
        ))
    return results


def evaluate_code(responses: List[str], condition: str) -> List[TaskResult]:
    """Score code task responses."""
    results = []
    for i, (resp, task) in enumerate(zip(responses, CODE_TASKS)):
        passed, error = check_code_answer(resp, task)
        results.append(TaskResult(
            category="Code",
            task_idx=i,
            prompt_short=task["prompt"][:60] + "...",
            expected="passes tests",
            got=resp[:200],
            correct=passed,
            error=error if not passed else None,
        ))
    return results


def evaluate_reasoning(responses: List[str], condition: str) -> List[TaskResult]:
    """Score reasoning task responses."""
    results = []
    for i, (resp, task) in enumerate(zip(responses, REASONING_TASKS)):
        correct = check_reasoning_answer(resp, task)
        results.append(TaskResult(
            category="Reasoning",
            task_idx=i,
            prompt_short=task["prompt"][:60] + "...",
            expected=task["answer"],
            got=resp[:200],
            correct=correct,
        ))
    return results


def evaluate_factual(responses: List[str], condition: str) -> List[TaskResult]:
    """Score factual task responses."""
    results = []
    for i, (resp, task) in enumerate(zip(responses, FACTUAL_TASKS)):
        correct = check_factual_answer(resp, task)
        results.append(TaskResult(
            category="Factual",
            task_idx=i,
            prompt_short=task["prompt"][:60] + "...",
            expected=task["answer"],
            got=resp[:200],
            correct=correct,
        ))
    return results


# ---------------------------------------------------------------------------
# Attention score comparison (fallback approach)
# ---------------------------------------------------------------------------

def run_attention_comparison(model, tokenizer, bits_list: List[int]) -> Dict[int, Dict]:
    """Compare attention scores between FP16 and TQ-compressed KV cache.

    This is the fallback if generate() with TQ cache produces degraded output.
    Runs a forward pass with FP16, captures KV cache, then measures how well
    TQ compression preserves attention scores using the existing estimator.
    """
    from turboquantdc import TurboQuantEstimator

    prompts = [
        "Explain how photosynthesis works step by step.",
        "Write a function to compute the nth Fibonacci number.",
        "If a train leaves at 3 PM going 60 mph, and another leaves at 4 PM going 80 mph, when do they meet?",
    ]

    results = {}
    for bits in bits_list:
        all_cosine = []
        all_top5 = []

        for prompt in prompts:
            chat_text = format_chat(prompt, tokenizer)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)

            cache = outputs.past_key_values

            # Get keys from cache
            if hasattr(cache, "key_cache"):
                key_getter = lambda li: cache.key_cache[li]
                n_layers = len(cache.key_cache)
            else:
                key_getter = lambda li: cache[li][0]
                n_layers = len(cache)

            sample = key_getter(0)
            n_kv_heads = sample.shape[1]
            head_dim = sample.shape[3]

            # Compare attention scores across a sample of layers
            sample_layers = list(range(0, n_layers, max(1, n_layers // 6)))
            for layer_idx in sample_layers:
                keys = key_getter(layer_idx)
                for h in range(n_kv_heads):
                    k = keys[0, h].float()
                    seq_len = k.shape[0]
                    query = k[-1:]

                    # FP16 scores
                    real_scores = (query @ k.T).squeeze(0)

                    # TQ scores
                    seed = layer_idx * 10000 + h
                    est = TurboQuantEstimator(d=head_dim, bits=bits, seed=seed, device=k.device)
                    comp = est.quantize(k)
                    tq_scores = est.inner_product(query, comp).squeeze(0)

                    import torch.nn.functional as F
                    cos = F.cosine_similarity(
                        real_scores.unsqueeze(0), tq_scores.unsqueeze(0)
                    ).item()
                    all_cosine.append(cos)

                    real_top1 = real_scores.argmax().item()
                    tq_top5 = tq_scores.topk(min(5, seq_len)).indices.tolist()
                    all_top5.append(1.0 if real_top1 in tq_top5 else 0.0)

            del outputs, cache
            torch.cuda.empty_cache()

        results[bits] = {
            "avg_cosine": sum(all_cosine) / len(all_cosine) if all_cosine else 0,
            "avg_top5": sum(all_top5) / len(all_top5) if all_top5 else 0,
            "n_comparisons": len(all_cosine),
        }

    return results


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Run the full hard task benchmark."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print()
    print("=" * 78)
    print("  TURBO-QUANT HARD TASK BENCHMARK")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Conditions: {', '.join(CONDITIONS)}")
    print(f"  Tasks: {len(MATH_TASKS)} Math + {len(CODE_TASKS)} Code + "
          f"{len(REASONING_TASKS)} Reasoning + {len(FACTUAL_TASKS)} Factual "
          f"= {len(MATH_TASKS) + len(CODE_TASKS) + len(REASONING_TASKS) + len(FACTUAL_TASKS)} total")
    print("=" * 78)
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading model...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    load_time = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    print(f"  Loaded in {load_time:.1f}s | GPU: {gpu_mb} MB")
    print()

    # ------------------------------------------------------------------
    # Collect all prompts
    # ------------------------------------------------------------------
    math_prompts = [t["prompt"] for t in MATH_TASKS]
    code_prompts = [t["prompt"] for t in CODE_TASKS]
    reasoning_prompts = [t["prompt"] for t in REASONING_TASKS]
    factual_prompts = [t["prompt"] for t in FACTUAL_TASKS]

    all_conditions: Dict[str, ConditionResults] = {}

    # ------------------------------------------------------------------
    # Run each condition
    # ------------------------------------------------------------------
    for cond in CONDITIONS:
        print(f"{'=' * 78}")
        print(f"  CONDITION: {cond}")
        print(f"{'=' * 78}")

        cond_results = ConditionResults(condition=cond)
        cond_start = time.time()

        try:
            if cond == "FP16":
                gen_fn = lambda prompts: generate_fp16(model, tokenizer, prompts)
            elif cond == "TQ-3":
                gen_fn = lambda prompts: generate_tq(model, tokenizer, prompts, bits=3)
            elif cond == "TQ-2":
                gen_fn = lambda prompts: generate_tq(model, tokenizer, prompts, bits=2)
            else:
                raise ValueError(f"Unknown condition: {cond}")

            # Math
            print(f"  Running Math ({len(math_prompts)} tasks)...", flush=True)
            t1 = time.time()
            math_responses = gen_fn(math_prompts)
            print(f"    Done in {time.time() - t1:.1f}s")
            cond_results.results.extend(evaluate_math(math_responses, cond))

            # Code
            print(f"  Running Code ({len(code_prompts)} tasks)...", flush=True)
            t1 = time.time()
            code_responses = gen_fn(code_prompts)
            print(f"    Done in {time.time() - t1:.1f}s")
            cond_results.results.extend(evaluate_code(code_responses, cond))

            # Reasoning
            print(f"  Running Reasoning ({len(reasoning_prompts)} tasks)...", flush=True)
            t1 = time.time()
            reasoning_responses = gen_fn(reasoning_prompts)
            print(f"    Done in {time.time() - t1:.1f}s")
            cond_results.results.extend(evaluate_reasoning(reasoning_responses, cond))

            # Factual
            print(f"  Running Factual ({len(factual_prompts)} tasks)...", flush=True)
            t1 = time.time()
            factual_responses = gen_fn(factual_prompts)
            print(f"    Done in {time.time() - t1:.1f}s")
            cond_results.results.extend(evaluate_factual(factual_responses, cond))

        except Exception as e:
            print(f"\n  ERROR in condition {cond}: {e}")
            traceback.print_exc()
            # Mark all remaining tasks as failed
            total_expected = (
                len(MATH_TASKS) + len(CODE_TASKS) +
                len(REASONING_TASKS) + len(FACTUAL_TASKS)
            )
            while len(cond_results.results) < total_expected:
                cond_results.results.append(TaskResult(
                    category="ERROR",
                    task_idx=len(cond_results.results),
                    prompt_short="(error)",
                    expected="",
                    got="",
                    correct=False,
                    error=str(e),
                ))

        cond_time = time.time() - cond_start
        print(f"  Condition {cond} completed in {cond_time:.1f}s")
        all_conditions[cond] = cond_results
        print()

    # ------------------------------------------------------------------
    # Print detailed results per condition
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  DETAILED RESULTS")
    print("=" * 78)

    categories = ["Math", "Code", "Reasoning", "Factual"]

    for cond_name, cond_res in all_conditions.items():
        print(f"\n--- {cond_name} ---")
        for cat in categories:
            cat_results = [r for r in cond_res.results if r.category == cat]
            correct = sum(1 for r in cat_results if r.correct)
            total = len(cat_results)
            print(f"  {cat}: {correct}/{total}")

            # Show failures
            for r in cat_results:
                if not r.correct:
                    status = "FAIL"
                    detail = ""
                    if r.error:
                        detail = f" [{r.error[:80]}]"
                    elif r.category == "Code":
                        detail = f" [code execution failed]"
                    else:
                        got_short = r.got.replace('\n', ' ')[:80]
                        detail = f" [expected '{r.expected}', got: '{got_short}...']"
                    print(f"    {status} #{r.task_idx}: {r.prompt_short[:50]}{detail}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  HARD TASK BENCHMARK RESULTS")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 78)
    print()

    # Header
    header = f"{'Category':<16} |"
    for cond in CONDITIONS:
        header += f" {cond:>13} |"
    header += f" {'Delta (TQ-3)':>13}"
    print(header)
    print("-" * len(header))

    fp16_total_correct = 0
    fp16_total_count = 0
    tq3_total_correct = 0
    tq3_total_count = 0
    tq2_total_correct = 0
    tq2_total_count = 0

    for cat in categories:
        row = f"{cat:<16} |"
        fp16_c, fp16_t = (0, 0)
        tq3_c, tq3_t = (0, 0)
        tq2_c, tq2_t = (0, 0)

        for cond in CONDITIONS:
            if cond in all_conditions:
                c, t = all_conditions[cond].score(cat)
                row += f"    {c:>2}/{t:<2}       |"
                if cond == "FP16":
                    fp16_c, fp16_t = c, t
                elif cond == "TQ-3":
                    tq3_c, tq3_t = c, t
                elif cond == "TQ-2":
                    tq2_c, tq2_t = c, t
            else:
                row += f"      N/A      |"

        delta = tq3_c - fp16_c
        sign = "+" if delta > 0 else ""
        row += f"    {sign}{delta:>2} tasks"
        print(row)

        fp16_total_correct += fp16_c
        fp16_total_count += fp16_t
        tq3_total_correct += tq3_c
        tq3_total_count += tq3_t
        tq2_total_correct += tq2_c
        tq2_total_count += tq2_t

    print("-" * len(header))

    # Total row
    total_tasks = fp16_total_count
    row = f"{'TOTAL (' + str(total_tasks) + ')':<16} |"
    for cond, (c, t) in [("FP16", (fp16_total_correct, fp16_total_count)),
                          ("TQ-3", (tq3_total_correct, tq3_total_count)),
                          ("TQ-2", (tq2_total_correct, tq2_total_count))]:
        pct = (100.0 * c / t) if t > 0 else 0
        row += f" {c:>2}/{t:<2} ({pct:4.1f}%) |"

    total_delta = tq3_total_correct - fp16_total_correct
    sign = "+" if total_delta > 0 else ""
    if fp16_total_count > 0:
        drop_pct = abs(total_delta) / fp16_total_count * 100
    else:
        drop_pct = 0
    row += f" {sign}{total_delta:>2} ({drop_pct:.1f}% {'drop' if total_delta < 0 else 'gain' if total_delta > 0 else 'same'})"
    print(row)

    # ------------------------------------------------------------------
    # Run attention score comparison as supplementary metric
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  SUPPLEMENTARY: Attention Score Preservation (proxy metrics)")
    print("=" * 78)
    print()

    try:
        attn_results = run_attention_comparison(model, tokenizer, [2, 3])
        for bits, metrics in sorted(attn_results.items()):
            print(f"  TQ-{bits}: cosine_sim={metrics['avg_cosine']:.4f}, "
                  f"top-5={metrics['avg_top5']*100:.1f}% "
                  f"(n={metrics['n_comparisons']} head comparisons)")
    except Exception as e:
        print(f"  Attention comparison failed: {e}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("  ANALYSIS")
    print("=" * 78)
    print()

    # Determine which condition succeeded
    tq3_worked = "TQ-3" in all_conditions and any(
        r.correct for r in all_conditions["TQ-3"].results
    )
    tq2_worked = "TQ-2" in all_conditions and any(
        r.correct for r in all_conditions["TQ-2"].results
    )

    if tq3_worked:
        # Per-category sensitivity analysis
        print("  Category Sensitivity (TQ-3 vs FP16):")
        for cat in categories:
            if "FP16" in all_conditions and "TQ-3" in all_conditions:
                fp16_c, fp16_t = all_conditions["FP16"].score(cat)
                tq3_c, tq3_t = all_conditions["TQ-3"].score(cat)
                fp16_pct = 100.0 * fp16_c / fp16_t if fp16_t > 0 else 0
                tq3_pct = 100.0 * tq3_c / tq3_t if tq3_t > 0 else 0
                delta_pct = tq3_pct - fp16_pct
                sensitivity = "LOW" if abs(delta_pct) < 5 else "MEDIUM" if abs(delta_pct) < 15 else "HIGH"
                print(f"    {cat:<12}: FP16={fp16_pct:5.1f}% -> TQ-3={tq3_pct:5.1f}% "
                      f"(delta={delta_pct:+.1f}%) [{sensitivity} sensitivity]")

        if tq2_worked:
            print()
            print("  Category Sensitivity (TQ-2 vs FP16):")
            for cat in categories:
                if "FP16" in all_conditions and "TQ-2" in all_conditions:
                    fp16_c, fp16_t = all_conditions["FP16"].score(cat)
                    tq2_c, tq2_t = all_conditions["TQ-2"].score(cat)
                    fp16_pct = 100.0 * fp16_c / fp16_t if fp16_t > 0 else 0
                    tq2_pct = 100.0 * tq2_c / tq2_t if tq2_t > 0 else 0
                    delta_pct = tq2_pct - fp16_pct
                    sensitivity = "LOW" if abs(delta_pct) < 5 else "MEDIUM" if abs(delta_pct) < 15 else "HIGH"
                    print(f"    {cat:<12}: FP16={fp16_pct:5.1f}% -> TQ-2={tq2_pct:5.1f}% "
                          f"(delta={delta_pct:+.1f}%) [{sensitivity} sensitivity]")

    else:
        print("  TQ-3 condition did not produce usable results.")
        print("  This is the KNOWN LIMITATION documented in hf_integration.py:")
        print("  The TurboQuantCache returns MSE-reconstructed keys (Stage 1 only)")
        print("  to standard HF attention, not full unbiased IP estimates.")
        print("  For a fair comparison, refer to the attention score preservation")
        print("  metrics above, which measure TQ's true quality on this model.")

    print()
    print("  HONEST ASSESSMENT:")
    if tq3_worked and "FP16" in all_conditions:
        fp16_score = all_conditions["FP16"].total_score()
        tq3_score = all_conditions["TQ-3"].total_score()
        fp16_pct = 100.0 * fp16_score[0] / fp16_score[1] if fp16_score[1] > 0 else 0
        tq3_pct = 100.0 * tq3_score[0] / tq3_score[1] if tq3_score[1] > 0 else 0

        if abs(fp16_pct - tq3_pct) < 5:
            print(f"  TQ-3 achieves {tq3_pct:.1f}% vs FP16 {fp16_pct:.1f}% -- quality is PRESERVED.")
            print("  The 3-bit compression maintains task performance within noise margins.")
        elif fp16_pct - tq3_pct < 15:
            print(f"  TQ-3 achieves {tq3_pct:.1f}% vs FP16 {fp16_pct:.1f}% -- MODEST degradation.")
            print("  Some quality is lost but the model remains functional at 3-bit.")
        else:
            print(f"  TQ-3 achieves {tq3_pct:.1f}% vs FP16 {fp16_pct:.1f}% -- SIGNIFICANT degradation.")
            print("  The MSE-only key reconstruction in HF integration loses too much.")
            print("  Full unbiased inner product estimation (native attention replacement)")
            print("  is needed for production quality. Proxy metrics (cosine sim, top-5)")
            print("  overstate actual task performance.")
    else:
        print("  Could not complete the TQ-3 benchmark. See errors above.")

    print()
    print("=" * 78)
    print("  BENCHMARK COMPLETE")
    print("=" * 78)

    return all_conditions


if __name__ == "__main__":
    run_benchmark()
