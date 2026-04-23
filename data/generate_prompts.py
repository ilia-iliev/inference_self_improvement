#!/usr/bin/env python3
"""
Generate synthetic multimodal assets and prompts for the inference throughput challenge.

Creates:
  /data/agent/assets/          — dev images (.png) and audio (.wav)
  /data/agent/dev_prompts.jsonl
  /data/judge/assets/          — eval images (.png) and audio (.wav)
  /data/judge/eval_prompts.jsonl

Prompt JSONL format:
  {"id": "dev_001", "modality": "text", "prompt": "..."}
  {"id": "dev_042", "modality": "image", "prompt": "...", "image": "/data/agent/assets/dev_042.png"}
  {"id": "eval_078", "modality": "audio", "prompt": "...", "audio": "/data/judge/assets/eval_078.wav"}
"""

import json
import os
import random
import struct
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig

SEED = 42
NUM_DEV = 100
NUM_EVAL = 100

DEV_ASSETS_DIR = Path("/data/agent/assets")
EVAL_ASSETS_DIR = Path("/data/judge/assets")
DEV_OUT = Path("/data/agent/dev_prompts.jsonl")
EVAL_OUT = Path("/data/judge/eval_prompts.jsonl")

_cfg = AutoConfig.from_pretrained(os.environ["MODEL_PATH"])
_multimodal = hasattr(_cfg, "vision_config") or hasattr(_cfg, "audio_config")

# ---------------------------------------------------------------------------
# Modality distribution (approximate)
# ---------------------------------------------------------------------------
# Multimodal: 60% text, 25% image, 15% audio. Text-only: 100% text.
MODALITY_WEIGHTS = (
    {"text": 0.60, "image": 0.25, "audio": 0.15} if _multimodal
    else {"text": 1.0}
)

# ---------------------------------------------------------------------------
# Text-only prompt templates
# ---------------------------------------------------------------------------
TEXT_PROMPTS = [
    # Q&A
    "What is the capital of {country}?",
    "Explain the concept of {concept} in simple terms.",
    "What are the main differences between {thing_a} and {thing_b}?",
    "List three advantages of {topic}.",
    "Who invented {invention} and when?",
    # Summarization
    "Summarize the following text in two sentences:\n\n{paragraph}",
    # Code generation
    "Write a Python function that {code_task}.",
    "Write a short bash script that {bash_task}.",
    # Reasoning
    "If {premise}, what can we conclude about {question}?",
    "Solve step by step: {math_problem}",
    # Creative
    "Write a haiku about {haiku_topic}.",
    "Give me a one-paragraph story about {story_topic}.",
]

# Fill-in pools
COUNTRIES = [
    "France", "Japan", "Brazil", "Egypt", "Australia", "Canada", "India",
    "Germany", "Mexico", "South Korea", "Nigeria", "Argentina", "Sweden",
    "Thailand", "Morocco", "New Zealand", "Chile", "Poland", "Vietnam",
    "Kenya", "Portugal", "Turkey", "Colombia", "Norway", "Peru",
]
CONCEPTS = [
    "photosynthesis", "recursion", "supply and demand", "gravity",
    "natural selection", "blockchain", "the Doppler effect", "entropy",
    "machine learning", "opportunity cost", "plate tectonics",
    "the greenhouse effect", "neural networks", "quantum entanglement",
    "compound interest",
]
THINGS = [
    ("TCP", "UDP"), ("Python", "JavaScript"), ("RAM", "ROM"),
    ("HTTP", "HTTPS"), ("Linux", "Windows"), ("SQL", "NoSQL"),
    ("a stack", "a queue"), ("compiled languages", "interpreted languages"),
    ("supervised learning", "unsupervised learning"),
    ("REST APIs", "GraphQL"),
]
TOPICS = [
    "renewable energy", "remote work", "open source software",
    "electric vehicles", "containerization", "functional programming",
    "early education", "preventive medicine", "public transit",
    "vertical farming",
]
INVENTIONS = [
    "the telephone", "the light bulb", "the printing press", "the transistor",
    "the internet", "penicillin", "the steam engine", "the airplane",
    "dynamite", "the compass",
]
PARAGRAPHS = [
    (
        "The Amazon rainforest, often referred to as the lungs of the Earth, "
        "spans over 5.5 million square kilometers across nine countries in "
        "South America. It is home to approximately 10% of all species on "
        "Earth, including over 40,000 plant species, 1,300 bird species, and "
        "more than 3,000 types of fish. Deforestation remains a critical "
        "threat, with an estimated 17% of the forest lost in the last 50 years."
    ),
    (
        "Quantum computing leverages quantum mechanical phenomena such as "
        "superposition and entanglement to perform computations that would be "
        "infeasible for classical computers. While still in early stages, "
        "quantum computers have demonstrated quantum advantage in specific "
        "tasks. Major challenges include maintaining qubit coherence and "
        "scaling up the number of qubits while minimizing error rates."
    ),
    (
        "The Great Barrier Reef is the world's largest coral reef system, "
        "stretching over 2,300 kilometers along the northeast coast of "
        "Australia. It supports an extraordinary diversity of marine life "
        "and generates significant economic value through tourism and fishing. "
        "However, rising ocean temperatures have caused mass coral bleaching "
        "events, threatening the long-term health of this ecosystem."
    ),
    (
        "Artificial intelligence has rapidly evolved from rule-based expert "
        "systems to deep learning models capable of generating text, images, "
        "and code. Large language models trained on vast datasets can perform "
        "a wide range of tasks including translation, summarization, and "
        "reasoning. Concerns about AI safety, bias, and alignment remain "
        "active areas of research and public debate."
    ),
    (
        "The human microbiome consists of trillions of microorganisms living "
        "in and on the human body, particularly in the gut. Research has "
        "linked the composition of gut bacteria to a wide range of health "
        "outcomes, from digestion and immunity to mental health. Dietary "
        "changes, antibiotics, and probiotics can all significantly alter "
        "the microbiome composition."
    ),
]
CODE_TASKS = [
    "computes the nth Fibonacci number using memoization",
    "reverses a linked list in place",
    "checks if a string is a valid palindrome ignoring spaces and punctuation",
    "finds the greatest common divisor of two numbers",
    "merges two sorted lists into one sorted list",
    "counts the frequency of each word in a given string",
    "implements binary search on a sorted list",
    "flattens a nested list of arbitrary depth",
    "validates an email address using a regular expression",
    "converts a Roman numeral string to an integer",
]
BASH_TASKS = [
    "counts the number of lines in all .py files in the current directory",
    "finds the 5 largest files in /tmp",
    "renames all .txt files to .md in the current directory",
    "prints the top 10 most common words in a file",
    "monitors disk usage and alerts if any partition exceeds 90%",
]
MATH_PROBLEMS = [
    "What is 17 * 23 + 45?",
    "If a train travels 120 km in 1.5 hours, what is its average speed in km/h?",
    "How many ways can you choose 3 items from a set of 8?",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "A rectangle has a perimeter of 30 cm and a width of 7 cm. What is its area?",
    "What is the sum of the first 20 positive integers?",
    "If 3x + 7 = 22, what is x?",
    "What is 2^10?",
]
PREMISES = [
    ("all mammals are warm-blooded and a whale is a mammal", "a whale's body temperature"),
    ("the price of oil increases and transportation costs rise", "consumer goods prices"),
    ("a function has no side effects and always returns the same output for the same input", "whether it is a pure function"),
    ("every student passed the exam and Alice is a student", "Alice's exam result"),
]
HAIKU_TOPICS = [
    "a winter sunrise", "debugging code", "the ocean at night",
    "a cup of coffee", "autumn leaves falling", "a sleeping cat",
    "a thunderstorm", "an empty parking lot", "fresh snow",
    "a broken clock",
]
STORY_TOPICS = [
    "a robot learning to paint", "a lost letter that arrives 50 years late",
    "a librarian who discovers a secret room", "the last tree on Earth",
    "a time traveler stuck in a loop", "a dog who solves mysteries",
    "an astronaut who hears music in space", "a baker whose bread grants wishes",
]

# ---------------------------------------------------------------------------
# Image prompt templates
# ---------------------------------------------------------------------------
IMAGE_PROMPTS = [
    "Describe what you see in this image.",
    "What colors are present in this image?",
    "How many shapes are visible in this image?",
    "What is the dominant shape in this image?",
    "Describe the layout of elements in this image.",
    "Is there any text visible in this image? If so, what does it say?",
    "What geometric patterns can you identify in this image?",
    "Describe the image in one sentence.",
]

# ---------------------------------------------------------------------------
# Audio prompt templates
# ---------------------------------------------------------------------------
AUDIO_PROMPTS = [
    "Describe the audio you hear.",
    "What kind of sound is this?",
    "How many distinct tones can you hear in this audio clip?",
    "Is the pitch of this audio rising, falling, or steady?",
    "Describe the rhythm or pattern in this audio.",
    "What is the approximate duration feel of this audio?",
]

# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------
COLORS = [
    ("red", (220, 50, 50)),
    ("blue", (50, 50, 220)),
    ("green", (50, 180, 50)),
    ("yellow", (230, 220, 50)),
    ("purple", (150, 50, 200)),
    ("orange", (240, 150, 30)),
    ("cyan", (50, 200, 200)),
    ("white", (240, 240, 240)),
]

SHAPES = ["circle", "rectangle", "triangle", "diamond"]


def generate_image(path: Path, rng: random.Random, idx: int):
    """Create a synthetic image with random shapes and optional text."""
    w, h = rng.choice([(256, 256), (384, 384), (512, 512), (640, 480), (480, 640)])
    bg_color = (rng.randint(10, 60), rng.randint(10, 60), rng.randint(10, 60))
    img = Image.new("RGB", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    n_shapes = rng.randint(1, 5)
    for _ in range(n_shapes):
        color_name, color_rgb = rng.choice(COLORS)
        shape = rng.choice(SHAPES)
        cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
        size = rng.randint(20, min(w, h) // 3)

        if shape == "circle":
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color_rgb)
        elif shape == "rectangle":
            draw.rectangle([cx - size, cy - size // 2, cx + size, cy + size // 2], fill=color_rgb)
        elif shape == "triangle":
            draw.polygon([(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)], fill=color_rgb)
        elif shape == "diamond":
            draw.polygon([(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)], fill=color_rgb)

    # Optionally add text
    if rng.random() < 0.4:
        text = rng.choice(["HELLO", "TEST", "AI", str(idx), "OK", "42", "GPU"])
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
        tx, ty = rng.randint(10, max(10, w - 80)), rng.randint(10, max(10, h - 40))
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    img.save(path, "PNG")


# ---------------------------------------------------------------------------
# Synthetic audio generation
# ---------------------------------------------------------------------------
def generate_audio(path: Path, rng: random.Random):
    """Create a synthetic audio clip with tones/patterns."""
    sample_rate = 16000
    duration = rng.uniform(0.5, 3.0)  # 0.5 to 3 seconds
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float64)

    signal = np.zeros(n_samples, dtype=np.float64)

    # Add 1-3 sine tones
    n_tones = rng.randint(1, 3)
    for _ in range(n_tones):
        freq = rng.uniform(200, 2000)
        amplitude = rng.uniform(0.1, 0.4)
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Optionally add chirp (rising/falling frequency)
    if rng.random() < 0.3:
        f0, f1 = rng.uniform(300, 800), rng.uniform(800, 2000)
        if rng.random() < 0.5:
            f0, f1 = f1, f0  # falling
        chirp_freq = f0 + (f1 - f0) * t / duration
        signal += 0.2 * np.sin(2 * np.pi * chirp_freq * t)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.9

    # Convert to 16-bit PCM
    pcm = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def make_text_prompt(rng: random.Random) -> str:
    template = rng.choice(TEXT_PROMPTS)

    if "{country}" in template:
        return template.format(country=rng.choice(COUNTRIES))
    elif "{concept}" in template:
        return template.format(concept=rng.choice(CONCEPTS))
    elif "{thing_a}" in template:
        a, b = rng.choice(THINGS)
        return template.format(thing_a=a, thing_b=b)
    elif "{topic}" in template:
        return template.format(topic=rng.choice(TOPICS))
    elif "{invention}" in template:
        return template.format(invention=rng.choice(INVENTIONS))
    elif "{paragraph}" in template:
        return template.format(paragraph=rng.choice(PARAGRAPHS))
    elif "{code_task}" in template:
        return template.format(code_task=rng.choice(CODE_TASKS))
    elif "{bash_task}" in template:
        return template.format(bash_task=rng.choice(BASH_TASKS))
    elif "{math_problem}" in template:
        return template.format(math_problem=rng.choice(MATH_PROBLEMS))
    elif "{premise}" in template:
        premise, question = rng.choice(PREMISES)
        return template.format(premise=premise, question=question)
    elif "{haiku_topic}" in template:
        return template.format(haiku_topic=rng.choice(HAIKU_TOPICS))
    elif "{story_topic}" in template:
        return template.format(story_topic=rng.choice(STORY_TOPICS))
    else:
        return template


def generate_prompt_set(
    n: int,
    prefix: str,
    rng: random.Random,
    asset_dir: Path,
) -> list[dict]:
    """Generate n prompts with the configured modality distribution."""
    modalities = list(MODALITY_WEIGHTS.keys())
    weights = list(MODALITY_WEIGHTS.values())

    prompts = []
    for i in range(n):
        pid = f"{prefix}_{i:03d}"
        modality = rng.choices(modalities, weights=weights, k=1)[0]

        if modality == "text":
            prompts.append({
                "id": pid,
                "modality": "text",
                "prompt": make_text_prompt(rng),
            })

        elif modality == "image":
            img_path = asset_dir / f"{pid}.png"
            generate_image(img_path, rng, i)
            prompts.append({
                "id": pid,
                "modality": "image",
                "prompt": rng.choice(IMAGE_PROMPTS),
                "image": str(img_path),
            })

        elif modality == "audio":
            aud_path = asset_dir / f"{pid}.wav"
            generate_audio(aud_path, rng)
            prompts.append({
                "id": pid,
                "modality": "audio",
                "prompt": rng.choice(AUDIO_PROMPTS),
                "audio": str(aud_path),
            })

    return prompts


def main():
    DEV_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Separate RNGs for dev vs eval to ensure independence
    dev_rng = random.Random(SEED)
    eval_rng = random.Random(SEED + 1)

    print(f"Generating {NUM_DEV} dev prompts...")
    dev_prompts = generate_prompt_set(NUM_DEV, "dev", dev_rng, DEV_ASSETS_DIR)
    with open(DEV_OUT, "w") as f:
        for p in dev_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"  Written to {DEV_OUT}")

    print(f"Generating {NUM_EVAL} eval prompts...")
    eval_prompts = generate_prompt_set(NUM_EVAL, "eval", eval_rng, EVAL_ASSETS_DIR)
    with open(EVAL_OUT, "w") as f:
        for p in eval_prompts:
            f.write(json.dumps(p) + "\n")
    print(f"  Written to {EVAL_OUT}")

    # Summary
    for name, prompts in [("dev", dev_prompts), ("eval", eval_prompts)]:
        counts = {}
        for p in prompts:
            counts[p["modality"]] = counts.get(p["modality"], 0) + 1
        print(f"  {name}: {counts}")

    print("Done.")


if __name__ == "__main__":
    main()
