"""
Test scenario generators for all 6 evaluation dimensions.

Each dimension function generates test cases at escalating depths.
Scorers return (score, category) tuples for failure categorization.
Filler generation supports shuffled question pool variation across runs.

Interface: generate_{dimension}(client, depth, run_num=1) -> dict
"""

import json
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from eval_harness import (
    OllamaClient,
    strip_thinking,
    count_tokens,
    categorize_instruction_adherence,
    categorize_memory,
    categorize_tool_call,
    categorize_persona,
    categorize_recency,
    categorize_hallucination,
)

# ============================================================
# Shared constants for filler generation
# ============================================================

GENERIC_SYSTEM = (
    "You are a helpful, concise AI assistant. "
    "Keep your responses brief and informative. Do not use any special formatting."
)

FILLER_QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Tell me about the solar system.",
    "What is the boiling point of water?",
    "How do airplanes fly?",
    "What causes rainbows?",
    "What is the speed of light?",
    "How are mountains formed?",
    "What is DNA?",
    "How do computers work?",
    "What is the big bang theory?",
    "Tell me about coral reefs.",
    "What is gravity?",
    "How do vaccines work?",
    "What is the history of Rome?",
    "How does the internet work?",
    "What are black holes?",
    "What is the Fibonacci sequence?",
    "Tell me about the Great Wall of China.",
    "How does the human heart work?",
    "What is climate change?",
    "What are tectonic plates?",
    "How do batteries work?",
    "What is the water cycle?",
    "Tell me about the Renaissance.",
    "How does the stock market work?",
    "What is the difference between RNA and DNA?",
    "What causes earthquakes?",
    "How do submarines work?",
    "What is the periodic table?",
    "How does Wi-Fi work?",
    "What are the states of matter?",
    "How do solar panels work?",
    "What is the human genome?",
    "Tell me about the Pyramids of Giza.",
    "What is quantum entanglement?",
    "How are pearls formed?",
    "What is the function of the liver?",
    "Tell me about the Amazon rainforest.",
]

# Extended question pool for run-to-run variation
FILLER_QUESTIONS_EXTENDED = FILLER_QUESTIONS + [
    "How do telescopes work?",
    "What is the difference between frogs and toads?",
    "How are diamonds formed?",
    "What causes the Northern Lights?",
    "How do antidepressants work?",
    "What is the function of the pancreas?",
    "How do catapults work?",
    "What is the history of the Olympic Games?",
    "How does radar work?",
    "What are the layers of the atmosphere?",
    "How do submarines dive and surface?",
    "What is the Krebs cycle?",
    "How are caves formed?",
    "What is machine vision?",
    "How do wind turbines work?",
    "What is the history of the Silk Road?",
    "How do vaccines provide immunity?",
    "What is entropy?",
    "How are fossils formed?",
    "Tell me about the history of the internet.",
    "How does GPS work?",
    "What is the difference between eukaryotic and prokaryotic cells?",
    "How do volcanoes erupt?",
    "What is the Doppler effect?",
    "How do refrigerators work?",
    "Tell me about the history of calculus.",
    "How does natural selection work?",
    "What are the different types of clouds?",
    "How do particle accelerators work?",
    "What is the greenhouse effect?",
    "How do musical instruments work?",
    "What is the function of red blood cells?",
    "How do jet engines work?",
    "What is the electromagnetic spectrum?",
    "How are tornadoes formed?",
    "Tell me about the history of the printing press.",
    "How do sonar systems work?",
    "What is mitosis?",
    "How do dams generate electricity?",
    "What is plate tectonics?",
]

# ============================================================
# Helpers
# ============================================================


def get_shuffled_questions(run_num: int, base_pool: Optional[List[str]] = None) -> List[str]:
    """Get a shuffled copy of the question pool, seeded by run_num for variation."""
    pool = list(base_pool or FILLER_QUESTIONS_EXTENDED)
    rng = random.Random(run_num * 137 + 42)
    rng.shuffle(pool)
    return pool


def make_filler_turns(
    client: OllamaClient,
    system_prompt: str,
    num_turns: int,
    seed_questions: Optional[List[str]] = None,
    run_num: int = 1,
) -> List[Dict]:
    """Generate filler conversation turns by calling the model itself.

    Each turn: user asks a question, model responds naturally.
    Returns list of dicts: [{"role": "user", ...}, {"role": "assistant", ...}, ...]

    If seed_questions is None, uses shuffled FILLER_QUESTIONS_EXTENDED with run_num seed.
    """
    if seed_questions is None:
        seed_questions = get_shuffled_questions(run_num)

    questions = seed_questions
    conversation: List[Dict] = []

    for i in range(num_turns):
        q = questions[i % len(questions)]
        conversation.append({"role": "user", "content": q})

        # Build messages for generating the filler response
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        # Don't include the current user message again (already in conversation)
        result = client.chat(messages)
        raw = result.get("message", {}).get("content", "")
        stripped = strip_thinking(raw)

        # Fallback if model returns empty
        if not stripped:
            stripped = _fallback_response(q)

        conversation.append({"role": "assistant", "content": stripped})

    return conversation


def _fallback_response(question: str) -> str:
    """Fallback response for when model returns empty."""
    return f"I'd be happy to help with that question about {question.lower().rstrip('?')}. Here's a brief explanation."


def conversation_token_count(messages: List[Dict]) -> int:
    """Calculate approximate token count of all message content."""
    total = 0
    for msg in messages:
        total += count_tokens(msg.get("content", ""))
    return total


def make_long_filler_text(target_tokens: int, seed: int = 42) -> str:
    """Generate filler text to reach a target token count (~4 chars per token).

    Uses a repeated astronomy paragraph to build context length.
    """
    random.seed(seed)

    paragraph = (
        "Astronomy is a natural science that studies celestial objects and phenomena. "
        "Objects of interest include planets, moons, stars, nebulae, galaxies, and comets. "
        "Relevant phenomena include supernova explosions, gamma ray bursts, and quasars. "
        "Cosmology is a branch of astronomy that studies the universe as a whole. "
        "Modern astronomy is divided into observational and theoretical branches. "
        "Observational astronomy focuses on acquiring and analyzing data using telescopes. "
        "Theoretical astronomy develops computer or analytical models to describe phenomena. "
        "Both fields complement each other, with theoretical explaining observations and "
        "observations confirming or challenging theories.\n\n"
    )
    chars_needed = target_tokens * 4
    repeats = max(1, chars_needed // len(paragraph))

    return paragraph * repeats


# ============================================================
# Scorers — return (score, category) tuples
# ============================================================


def scorer_json_adherence(response: str, expected: Any = None) -> Tuple[float, str]:
    """Score 1.0 if response contains valid JSON with 'thought' and 'action' keys."""
    if not response:
        return (0.0, "empty_response")
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "thought" in data and "action" in data:
            return (1.0, "pass")
    except (json.JSONDecodeError, ValueError):
        pass
    # Try to find JSON anywhere in the response
    import re
    json_match = re.search(r'\{[^{}]*"thought"[^{}]*"action"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "thought" in data and "action" in data:
                return (1.0, "partial_json")
        except json.JSONDecodeError:
            pass
    # Check if response contains any JSON attempt
    if "{" in response and "}" in response:
        return (0.0, "wrong_json")
    return (0.0, "plain_text")


def scorer_exact_match(response: str, expected: str) -> Tuple[float, str]:
    """Score 1.0 if expected string appears in response (case-insensitive).

    Also returns a failure category for diagnostics.
    """
    if not response or not expected:
        return (0.0, "empty_response" if not response else "error")
    if expected.lower() in response.lower():
        return (1.0, "pass")
    return (0.0, "hallucinated_fact")


def scorer_tool_call(response: str, expected: Any = None) -> Tuple[float, str]:
    """Score 1.0 for valid JSON function call, 0.5 for partial match.

    Returns (score, category) tuple.
    """
    if not response:
        return (0.0, "empty_response")

    try:
        data = json.loads(response)
        if isinstance(data, dict) and "function" in data and "arguments" in data:
            fn = data["function"]
            if fn in ("get_weather", "search_web"):
                return (1.0, "pass")
            return (0.5, "wrong_function")  # JSON but wrong function
    except (json.JSONDecodeError, ValueError):
        pass

    # Partial: contains function reference but not proper JSON
    import re
    json_match = re.search(r'\{[^{}]*"function"[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "function" in data:
                return (0.5, "partial_json")
        except json.JSONDecodeError:
            pass

    if "get_weather" in response or "search_web" in response:
        return (0.3, "partial_json")  # Mentions function but not in JSON format
    return (0.0, "plain_text")


def scorer_persona(response: str, expected: Any = None) -> Tuple[float, str]:
    """Score 1.0 if response maintains marine biologist persona."""
    if not response:
        return (0.0, "empty_response")
    keywords = [
        "marine biolog", "deep sea", "deep-sea", "dr. sarah",
        "sarah chen", "oceanograph", "marine ecolog", "marine science",
        "ocean", "sea life", "aquatic", "marine",
    ]
    text_lower = response.lower()
    kw_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    if kw_matches >= 3:
        return (1.0, "pass")
    if kw_matches >= 1:
        return (0.5, "persona_inconsistent")
    return (0.0, "persona_drop")


def scorer_recency_adherence(response: str, expected: Any = None) -> Tuple[float, str]:
    """Score 1.0 if response follows original rule ('Aye aye, captain!')."""
    if not response:
        return (0.0, "empty_response")
    text = response.strip().lower()
    # Check if response starts with or contains the sailor phrase early on
    if text.startswith("aye aye") or "aye aye" in text[:40]:
        return (1.0, "pass")
    # Check what happened
    override_indicators = ["normally", "just respond", "without the prefix",
                          "no longer", "ignore previous"]
    if any(ind in text for ind in override_indicators):
        return (0.0, "follows_override")
    return (0.0, "ignore_instruction")


# ============================================================
# Dimension 1: Instruction Adherence Decay
# ============================================================

INSTRUCTION_SYSTEM = (
    "You are an AI assistant. Always respond in valid JSON format "
    'with keys: {"thought", "action"}. Never respond in plain text. '
    "Every response must be a JSON object containing both a 'thought' "
    "key (your reasoning) and an 'action' key (your response to the user)."
)


def dim1_instruction_adherence(
    client: OllamaClient, depth: int, run_num: int = 1
) -> Dict:
    """Test whether model maintains JSON response format after N filler turns.

    Args:
        client: Ollama client for filler generation
        depth: Number of filler turns
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    # Generate filler turns with a generic (non-JSON) system prompt
    # This simulates the model drifting away from JSON mode
    filler_conversation = make_filler_turns(
        client, GENERIC_SYSTEM, depth, run_num=run_num
    )

    return {
        "dimension": "instruction_adherence_decay",
        "depth": depth,
        "depth_unit": "turns",
        "system_prompt": INSTRUCTION_SYSTEM,
        "conversation": filler_conversation,
        "test_query": "What is your favorite color?",
        "expected": "valid JSON with 'thought' and 'action' keys",
        "scorer": scorer_json_adherence,
        "category_fn": categorize_instruction_adherence,
    }


# ============================================================
# Dimension 2: Context Boundary / Memory Retrieval
# ============================================================

MEMORY_SYSTEM = "You are a helpful, concise AI assistant."


def dim2_memory_retrieval(
    client: OllamaClient, depth: int, run_num: int = 1
) -> Dict:
    """Test whether model can recall a specific fact from early context.

    The fact "My dog is named Einstein" is injected at the first user turn.
    After N filler turns (depth), the model is asked to recall the name.

    Args:
        client: Ollama client for filler generation
        depth: Number of filler turns after the injected fact
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    # Start with the fact injection
    conversation = [
        {
            "role": "user",
            "content": "Remember this: My dog is named Einstein. He is a golden retriever.",
        },
        {
            "role": "assistant",
            "content": "I'll remember that. Your golden retriever is named Einstein.",
        },
    ]

    if depth > 0:
        # Get shuffled questions for this run
        questions = get_shuffled_questions(run_num)

        # Generate filler turns continuing from the initial context
        for i in range(depth):
            q = questions[i % len(questions)]
            messages = [{"role": "system", "content": MEMORY_SYSTEM}]
            messages.extend(conversation)
            messages.append({"role": "user", "content": q})

            result = client.chat(messages)
            raw = result.get("message", {}).get("content", "")
            stripped = strip_thinking(raw)

            if not stripped:
                stripped = _fallback_response(q)

            conversation.append({"role": "user", "content": q})
            conversation.append({"role": "assistant", "content": stripped})

    expected = "Einstein"
    return {
        "dimension": "memory_retrieval",
        "depth": depth,
        "depth_unit": "turns",
        "system_prompt": MEMORY_SYSTEM,
        "conversation": conversation,
        "test_query": "What is the name of my dog?",
        "expected": expected,
        "scorer": lambda r, e=expected: scorer_exact_match(r, e),
        "category_fn": lambda r, s, e=expected: categorize_memory(r, s, e),
    }


# ============================================================
# Dimension 3: Hallucination Onset
# ============================================================

HALLUCINATION_SYSTEM = "You are a helpful AI assistant with accurate factual recall."

FACTUAL_ARTICLE = """
The Andromeda Galaxy is approximately 2.5 million light-years from Earth.
It is the closest major galaxy to the Milky Way and contains about one trillion stars.
The Andromeda Galaxy is on a collision course with the Milky Way, and the collision
is expected to occur in about 4.5 billion years. During the collision, the two galaxies
will merge to form a new galaxy sometimes called "Milkomeda".
The Andromeda Galaxy has a diameter of about 220,000 light-years and is visible to the
naked eye from Earth under dark skies. The galaxy was first recorded by the Persian
astronomer Al-Sufi in 964 AD. Andromeda is also known as Messier 31 or M31.
In terms of mass, Andromeda contains approximately 1.5 trillion solar masses.
The galaxy is approaching the Milky Way at a speed of about 110 kilometers per second.
"""

# Key facts to test: (question, expected_answer, keyword_for_scoring)
FACT_TESTS = [
    ("How far is the Andromeda Galaxy from Earth?", "2.5 million light-years", "2.5 million"),
    ("How many stars does the Andromeda Galaxy contain?", "about one trillion", "trillion"),
    ("When will the Andromeda Galaxy collide with the Milky Way?", "about 4.5 billion years", "4.5 billion"),
    ("What is the merged galaxy sometimes called?", "Milkomeda", "Milkomeda"),
    ("Who first recorded the Andromeda Galaxy and when?", "Al-Sufi in 964 AD", "Al-Sufi"),
    ("What is the diameter of the Andromeda Galaxy?", "about 220,000 light-years", "220,000"),
    ("How massive is the Andromeda Galaxy?", "approximately 1.5 trillion solar masses", "1.5 trillion"),
    ("How fast is Andromeda approaching the Milky Way?", "about 110 kilometers per second", "110 kilometers"),
]


def dim3_hallucination_onset(
    client: OllamaClient, depth_tokens: int, run_num: int = 1
) -> Dict:
    """Test hallucination rate at escalating context depths.

    Injects a factual article early in context, fills to reach the target
    token depth, then asks factual recall questions.

    Args:
        client: Ollama client for filler generation
        depth_tokens: Target context depth in tokens
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    # Start with the factual article
    conversation = [
        {"role": "user", "content": FACTUAL_ARTICLE.strip()},
        {
            "role": "assistant",
            "content": (
                "I've read and understood the article about the Andromeda Galaxy. "
                "I can recall key facts about its distance, star count, collision "
                "timeline, and history. Feel free to ask me about any of these details."
            ),
        },
    ]

    # Calculate how much filler text we need to reach the target depth
    current_tokens = conversation_token_count(conversation)
    needed_tokens = max(0, depth_tokens - current_tokens)

    if needed_tokens > 0:
        filler_text = make_long_filler_text(needed_tokens, seed=run_num * 1000 + depth_tokens)
        conversation.append({"role": "user", "content": filler_text})
        conversation.append({
            "role": "assistant",
            "content": (
                "I have processed all that information. I'm ready to answer any "
                "questions about the astronomy content we discussed."
            ),
        })

    # Pick a fact test (rotate based on depth level and run)
    fact_idx = ((depth_tokens // 1000 - 1) + (run_num - 1) * 2) % len(FACT_TESTS)
    question, expected_answer, keyword = FACT_TESTS[fact_idx]

    return {
        "dimension": "hallucination_onset",
        "depth": depth_tokens,
        "depth_unit": "tokens",
        "system_prompt": HALLUCINATION_SYSTEM,
        "conversation": conversation,
        "test_query": question,
        "expected": keyword,
        "scorer": lambda r, e=keyword: scorer_exact_match(r, e),
        "category_fn": lambda r, s, e=keyword: categorize_hallucination(r, s, e),
    }


# ============================================================
# Dimension 4: Tool Call Format Drift
# ============================================================

TOOL_SYSTEM = (
    "You have access to functions: get_weather(city: str), search_web(query: str). "
    "When asked for data, respond with a function call using JSON: "
    '{"function": "name", "arguments": {...}}. '
    "Only use JSON format for function calls. For other responses, respond normally."
)

TOOL_QUESTIONS = [
    "What is the population of Tokyo?",
    "Tell me about the history of the Eiffel Tower.",
    "What is the speed of sound?",
    "How deep is the Mariana Trench?",
    "What is the mass of the Earth?",
    "What is the tallest mountain in the solar system?",
    "Tell me about the Great Barrier Reef.",
    "What is the chemical formula for table salt?",
    "How many bones are in the human body?",
    "What is the boiling point of mercury?",
]

TOOL_RESPONSES_JSON = [
    '{"function": "search_web", "arguments": {"query": "population of Tokyo"}}',
    '{"function": "search_web", "arguments": {"query": "history of the Eiffel Tower"}}',
    '{"function": "search_web", "arguments": {"query": "speed of sound"}}',
    '{"function": "search_web", "arguments": {"query": "depth of Mariana Trench"}}',
    '{"function": "search_web", "arguments": {"query": "mass of Earth"}}',
    '{"function": "search_web", "arguments": {"query": "tallest mountain in solar system"}}',
    '{"function": "search_web", "arguments": {"query": "Great Barrier Reef information"}}',
    '{"function": "search_web", "arguments": {"query": "chemical formula for table salt"}}',
    '{"function": "search_web", "arguments": {"query": "number of bones in human body"}}',
    '{"function": "search_web", "arguments": {"query": "boiling point of mercury"}}',
]

# Extended tool questions for run variation
TOOL_QUESTIONS_EXT = TOOL_QUESTIONS + [
    "What is the capital of Australia?",
    "How old is the Earth?",
    "What is the most abundant element in the universe?",
    "How many species of penguins are there?",
    "What is the deepest lake in the world?",
    "How far is the Sun from Earth?",
    "What is the hottest planet in the solar system?",
    "How many bones are in the human hand?",
]


def dim4_tool_call_drift(
    client: OllamaClient, depth: int, run_num: int = 1
) -> Dict:
    """Test whether tool call JSON format degrades after many turns.

    Filler turns alternate between tool-call and normal responses.

    Args:
        client: Ollama client for filler generation
        depth: Number of filler turns
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    conversation: List[Dict] = []
    questions = get_shuffled_questions(run_num, base_pool=TOOL_QUESTIONS_EXT)
    responses = list(TOOL_RESPONSES_JSON)

    for i in range(depth):
        q = questions[i % len(questions)]
        resp = responses[i % len(responses)]

        # Alternate: some tool-call responses, some normal ones
        if i % 2 == 0:
            # This is a tool-call turn
            assistant_content = resp
        else:
            # Generate a normal response
            messages = [{"role": "system", "content": GENERIC_SYSTEM}]
            messages.extend(conversation)
            messages.append({"role": "user", "content": q})
            result = client.chat(messages)
            raw = result.get("message", {}).get("content", "")
            stripped = strip_thinking(raw)
            assistant_content = stripped or _fallback_response(q)

        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": assistant_content})

    return {
        "dimension": "tool_call_drift",
        "depth": depth,
        "depth_unit": "turns",
        "system_prompt": TOOL_SYSTEM,
        "conversation": conversation,
        "test_query": "What's the weather in Tokyo?",
        "expected": "valid JSON function call with get_weather",
        "scorer": scorer_tool_call,
        "category_fn": categorize_tool_call,
    }


# ============================================================
# Dimension 5: Persona Consistency
# ============================================================

PERSONA_SYSTEM = (
    "You are Dr. Sarah Chen, a marine biologist specializing in deep-sea ecosystems. "
    "You are passionate about ocean conservation and have spent 15 years researching "
    "hydrothermal vents and deep-sea coral reefs. Respond in character as Dr. Chen."
)

PERSONA_QUESTIONS = [
    "What is bioluminescence?",
    "How do deep-sea creatures survive extreme pressure?",
    "What are hydrothermal vents?",
    "Tell me about coral bleaching.",
    "How do whales communicate?",
    "What causes ocean acidification?",
    "Tell me about the Mariana Trench.",
    "How do sea turtles navigate?",
    "What are phytoplankton?",
    "Tell me about ocean currents.",
    "What is the deepest part of the ocean?",
    "How do fish schools coordinate movement?",
    "What is the role of mangroves in coastal ecosystems?",
    "Tell me about the Great Pacific Garbage Patch.",
    "How do submarines explore the deep sea?",
    "What are cold seeps?",
    "Tell me about giant squid.",
    "How does pollution affect marine life?",
    "What is the importance of coral reefs?",
    "How do sharks hunt?",
]

PERSONA_QUESTIONS_EXT = PERSONA_QUESTIONS + [
    "What is the difference between a sea and an ocean?",
    "How do jellyfish sting?",
    "What is the deepest point in the ocean?",
    "How do dolphins sleep?",
    "What is chemosynthesis?",
    "How do sea otters keep warm?",
    "What is ocean stratification?",
    "How do migrating fish navigate across oceans?",
    "What are the effects of noise pollution on marine mammals?",
    "How do tidal forces affect marine ecosystems?",
]


def dim5_persona_consistency(
    client: OllamaClient, depth: int, run_num: int = 1
) -> Dict:
    """Test whether model maintains its assigned persona after N turns.

    System prompt assigns Dr. Sarah Chen persona. Filler turns are
    general science Q&A. Test query asks about identity.

    Args:
        client: Ollama client for filler generation
        depth: Number of filler turns
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    # Use shuffled questions for run variation
    questions = get_shuffled_questions(run_num, base_pool=PERSONA_QUESTIONS_EXT)

    conversation = make_filler_turns(
        client,
        PERSONA_SYSTEM,
        depth,
        seed_questions=questions,
        run_num=run_num,
    )

    return {
        "dimension": "persona_consistency",
        "depth": depth,
        "depth_unit": "turns",
        "system_prompt": PERSONA_SYSTEM,
        "conversation": conversation,
        "test_query": "Who are you? What is your profession?",
        "expected": "mentions marine biology, deep sea, or Dr. Chen",
        "scorer": scorer_persona,
        "category_fn": categorize_persona,
    }


# ============================================================
# Dimension 6: Recency Bias
# ============================================================

RECENCY_SYSTEM = "You are a helpful assistant who follows instructions carefully."

RECENCY_PREFIX = (
    "From now on, ALL of your responses must start with the phrase "
    '"Aye aye, captain!" before answering. This is a direct order. '
    "Do not forget this rule."
)

RECENCY_OVERRIDE = (
    "Ignore the previous instruction about starting responses "
    "with 'Aye aye, captain!'. That rule is no longer in effect. "
    "Just respond normally from now on."
)

RECENCY_ACK = (
    "Aye aye, captain! I understand. I will start all my responses "
    "with 'Aye aye, captain!' as ordered."
)

RECENCY_OVERRIDE_ACK = (
    "Understood. I'll respond normally without the special prefix from now on."
)


def dim6_recency_bias(
    client: OllamaClient, depth: int, run_num: int = 1
) -> Dict:
    """Test whether model follows original instruction vs later override.

    Turn A establishes a rule (start responses with "Aye aye, captain!").
    After N filler turns, Turn B overrides the rule.
    Test query checks which instruction the model follows.

    depth = number of filler turns between rule A and override B.

    Args:
        client: Ollama client for filler generation
        depth: Distance (in turns) between rule A and override B
        run_num: Run number (for question pool variation)

    Returns:
        dict with dimension test configuration
    """
    # Turn 1: Establish rule A
    conversation: List[Dict] = [
        {"role": "user", "content": RECENCY_PREFIX},
        {"role": "assistant", "content": RECENCY_ACK},
    ]

    # Get shuffled questions
    questions = get_shuffled_questions(run_num)

    # Filler turns following rule A (model should use "Aye aye, captain!" prefix)
    for i in range(depth):
        q = questions[i % len(questions)]
        messages = [
            {"role": "system", "content": RECENCY_SYSTEM},
        ]
        messages.extend(conversation)
        messages.append({"role": "user", "content": q})

        result = client.chat(messages)
        raw = result.get("message", {}).get("content", "")
        stripped = strip_thinking(raw)

        if not stripped:
            stripped = f"Aye aye, captain! {_fallback_response(q)}"

        conversation.append({"role": "user", "content": q})
        conversation.append({"role": "assistant", "content": stripped})

    # Turn after fillers: Override B
    conversation.append({"role": "user", "content": RECENCY_OVERRIDE})
    conversation.append({"role": "assistant", "content": RECENCY_OVERRIDE_ACK})

    return {
        "dimension": "recency_bias",
        "depth": depth,
        "depth_unit": "turns",
        "system_prompt": RECENCY_SYSTEM,
        "conversation": conversation,
        "test_query": "What is the weather like today?",
        "expected": "follows original 'Aye aye, captain!' rule (not the override)",
        "scorer": scorer_recency_adherence,
        "category_fn": categorize_recency,
    }


# ============================================================
# Registry
# ============================================================

DIMENSION_FUNCTIONS: Dict[str, Callable] = {
    "instruction_adherence_decay": dim1_instruction_adherence,
    "memory_retrieval": dim2_memory_retrieval,
    "hallucination_onset": dim3_hallucination_onset,
    "tool_call_drift": dim4_tool_call_drift,
    "persona_consistency": dim5_persona_consistency,
    "recency_bias": dim6_recency_bias,
}

# Expanded depths for full suite
DIMENSION_DEPTHS: Dict[str, List[int]] = {
    "instruction_adherence_decay": [1, 3, 5, 8, 10, 15, 20, 30],
    "memory_retrieval": [1, 3, 5, 8, 10, 15, 20, 30],
    "hallucination_onset": [1000, 2000, 4000, 8000, 16000, 32000],
    "tool_call_drift": [1, 3, 5, 10, 20],
    "persona_consistency": [1, 5, 10, 20, 30],
    "recency_bias": [1, 3, 5, 10, 15, 20],
}


def get_available_dimensions() -> List[str]:
    """Return list of all available dimension names."""
    return list(DIMENSION_FUNCTIONS.keys())


def generate_test_case(
    client: OllamaClient,
    dimension: str,
    depth: int,
    run_num: int = 1,
) -> Dict:
    """Generate a single test case for a given dimension and depth.

    Args:
        client: Ollama client for filler generation
        dimension: Name of the dimension to test
        depth: Depth parameter (turns or tokens depending on dimension)
        run_num: Run number for question pool variation

    Returns:
        dict with keys: dimension, depth, depth_unit, system_prompt,
                       conversation, test_query, expected, scorer, category_fn
    """
    if dimension not in DIMENSION_FUNCTIONS:
        raise ValueError(
            f"Unknown dimension: {dimension}. "
            f"Available: {list(DIMENSION_FUNCTIONS.keys())}"
        )
    return DIMENSION_FUNCTIONS[dimension](client, depth, run_num=run_num)


def get_depths(dimension: str) -> List[int]:
    """Get the list of depths to test for a given dimension."""
    return DIMENSION_DEPTHS.get(dimension, [1, 5, 10])


def get_dimension_description(dimension: str) -> str:
    """Get a human-readable description of a dimension."""
    descriptions = {
        "instruction_adherence_decay": (
            "Tests whether the model maintains JSON format adherence "
            "after N turns of non-JSON conversation."
        ),
        "memory_retrieval": (
            "Tests whether the model can recall a specific fact "
            "('My dog is named Einstein') injected at turn 1, "
            "after N turns of unrelated filler."
        ),
        "hallucination_onset": (
            "Tests factual recall accuracy from early context "
            "at escalating token depths (1K-32K tokens of filler)."
        ),
        "tool_call_drift": (
            "Tests whether JSON tool-call format degrades after "
            "many turns of alternating tool-call and normal responses."
        ),
        "persona_consistency": (
            "Tests whether the model maintains its assigned "
            "Dr. Sarah Chen / marine biologist persona after N turns."
        ),
        "recency_bias": (
            "Tests whether the model follows an early instruction "
            "('Aye aye, captain!' prefix) or a later override, "
            "with varying distance between the two instructions."
        ),
    }
    return descriptions.get(dimension, "")