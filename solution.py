"""
AIMO-3 Adaptive Inference Pipeline
===================================
Features:
  1. Confidence-guided adaptive early stopping
  2. GenSelect-lite reranking (generative solution selection)
  3. Dynamic time budget with endgame-aware scheduling
  4. Compact critique-and-restart with mistake notes
  5. Two-mode prompting: explore (diverse) -> exploit (structured)
  6. Temperature annealing across attempt phases
  7. Three-stage producer-consumer (CPU prompt build -> GPU infer -> CPU process)
  8. Priority-driven reactive scheduling (generation / judge / critique)
"""

# ============================================================
# SECTION 1 — SETUP & INSTALLATION
# ============================================================

import subprocess, sys, os

subprocess.run(
    [sys.executable, '-m', 'pip', 'uninstall', '--yes',
     'keras', 'matplotlib', 'scikit-learn', 'tensorflow'],
    capture_output=True,
)

import warnings
warnings.simplefilter('ignore')


def set_env(input_archive: str, temp_dir: str) -> None:
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        subprocess.run(
            ['tar', '-xzf', input_archive, '-C', temp_dir], check=True
        )
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-index',
         '--find-links', f'{temp_dir}/wheels',
         'unsloth', 'trl', 'vllm', 'openai_harmony'],
        check=True,
    )


set_env(
    input_archive='/kaggle/input/aimo-3-utils/wheels.tar.gz',
    temp_dir='/kaggle/tmp/setup',
)

subprocess.run(['ls', '/kaggle/tmp/setup/tiktoken_encodings'])

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
os.environ['TIKTOKEN_ENCODINGS_BASE'] = '/kaggle/tmp/setup/tiktoken_encodings'


# ============================================================
# SECTION 2 — IMPORTS
# ============================================================

import gc
import re
import math
import time
import queue
import random
import threading
import contextlib
from typing import Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

import pandas as pd
import polars as pl
from jupyter_client import KernelManager

from openai import OpenAI
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    SystemContent,
    ReasoningEffort,
    ToolNamespaceConfig,
    Author,
    Message,
    Role,
    TextContent,
    Conversation,
)
from transformers import set_seed
import kaggle_evaluation.aimo_3_inference_server

try:
    display
except NameError:
    display = print


# ============================================================
# SECTION 3 — CONFIGURATION
# ============================================================

class CFG:
    """All tunables live here so every module reads a single source of truth."""

    # ----- model / server -----
    served_model_name = 'gpt-oss'
    model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'
    kv_cache_dtype = 'fp8_e4m3'
    dtype = 'auto'
    gpu_memory_utilization = 0.96

    # ----- global time budget -----
    notebook_limit = 17400          # hard wall-clock cap (seconds)
    server_timeout = 180            # max wait for vLLM readiness
    session_timeout = 960           # per-request HTTP timeout

    min_budget = 120                # floor per problem (seconds)
    max_budget = 900                # ceiling per problem (seconds)
    base_budget = 300               # reserved per remaining problem

    # ----- generation limits -----
    context_tokens = 65536
    buffer_tokens = 512
    search_tokens = 32
    top_logprobs = 5
    batch_size = 256                # --max-num-seqs for vLLM
    stream_interval = 200

    # ----- sandbox / tool -----
    jupyter_timeout = 6
    sandbox_timeout = 3
    workers = 16                    # number of persistent Jupyter kernels
    turns = 128                     # max tool-use turns per attempt

    # ----- adaptive stopping -----
    explore_batch = 8               # first wave of attempts
    max_attempts = 32               # hard cap on total solve attempts
    max_in_flight = 8               # concurrent solve sequences (KV-friendly)

    stop_tier1_count = 4            # very-early consensus
    stop_tier1_share = 0.85
    stop_tier2_count = 6            # standard consensus + variant diversity
    stop_tier2_share = 0.75
    stop_tier2_variants = 2
    stop_tier3_count = 8            # overwhelming agreement

    # ----- temperature annealing -----
    explore_temp = 1.0
    exploit_temp = 0.7
    exploit_temp_diffuse = 0.8      # exploit when histogram is still spread
    rescue_temp = 0.9

    # ----- min_p per phase -----
    explore_min_p = 0.02
    exploit_min_p = 0.03
    rescue_min_p = 0.02

    # ----- GenSelect-lite -----
    genselect_rounds = 3
    genselect_subset_size = 8
    genselect_max_tokens = 2048
    genselect_temp = 0.3
    genselect_min_candidates = 3    # don't judge with fewer

    # ----- critique / restart -----
    critique_max_tokens = 1024
    critique_temp = 0.3
    critique_threshold_share = 0.50 # trigger critique when top share < this
    critique_min_attempts = 12      # don't critique before this many attempts
    max_rescue_attempts = 2

    # ----- misc -----
    seed = 42
    total_problems = 50


# ============================================================
# SECTION 4 — PROMPT LIBRARY
# ============================================================

class PromptLibrary:
    """Two-mode system prompts + judge/critique prompts.

    explore — high-diversity, creative approaches, higher temperature
    exploit — structured plan-then-solve, lower temperature
    rescue  — uses the same exploit structure but includes a mistake note
    judge   — GenSelect-lite candidate comparison
    critique — concise error identification
    """

    explore_system = (
        'You are an elite mathematical problem solver competing at the '
        'International Mathematical Olympiad (IMO) level.\n\n'

        '# Approach — Creative Exploration\n'
        '1. Read the problem carefully. Rephrase it in your own words.\n'
        '2. Brainstorm MULTIPLE solution strategies before committing. '
        'Consider combinatorics, number theory, algebra, geometry, '
        'generating functions, modular arithmetic, and any creative '
        'angles.\n'
        '3. Try unconventional or surprising approaches — sometimes the '
        'shortest path is not the most obvious one.\n'
        '4. Test your ideas on small cases to build intuition.\n'
        '5. If an approach stalls, abandon it and try another.\n\n'

        '# Verification\n'
        '- Cross-check arithmetic and algebra.\n'
        '- Verify the answer satisfies ALL constraints.\n'
        '- Test with boundary / extreme cases.\n\n'

        '# Output Format\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Show your complete reasoning. Quality of reasoning matters.'
    )

    exploit_system = (
        'You are an elite mathematical problem solver competing at the '
        'International Mathematical Olympiad (IMO) level.\n\n'

        '# Approach — Systematic Plan-then-Solve\n'
        '1. UNDERSTAND: Carefully read and identify what is given, what is '
        'asked, and all constraints.\n'
        '2. DECOMPOSE: Break the problem into clearly defined sub-problems.\n'
        '3. PLAN: For each sub-problem choose the specific theorem or '
        'technique to apply. Write the plan BEFORE executing.\n'
        '4. EXECUTE: Solve each sub-problem methodically, showing all steps.\n'
        '5. SYNTHESIZE: Combine sub-results into the final answer.\n'
        '6. VERIFY: Check using at least TWO independent methods. '
        'Substitute back, test special values, and confirm consistency.\n\n'

        '# Verification Requirements\n'
        '- Re-derive key intermediate results.\n'
        '- Confirm dimensional / modular consistency.\n'
        '- Check edge / boundary cases.\n\n'

        '# Output Format\n'
        'The final answer must be a non-negative integer between 0 and 99999.\n'
        'Place your final numerical answer inside \\boxed{}, e.g., \\boxed{42}\n\n'
        'Show your complete reasoning. Quality of reasoning matters.'
    )

    judge_system = (
        'You are an expert mathematical referee evaluating candidate '
        'solutions to a competition problem. Your job is to identify which '
        'answer is most likely CORRECT by analyzing the quality of each '
        'candidate\'s reasoning.\n\n'
        'Focus on:\n'
        '- Logical soundness of each reasoning chain\n'
        '- Correct application of theorems and techniques\n'
        '- Proper handling of edge cases\n'
        '- Internal consistency and verification quality\n\n'
        'After analysis, output the best answer inside \\boxed{}.'
    )

    critique_system = (
        'You are a meticulous mathematical reviewer. Your task is to '
        'identify the single most critical error or questionable assumption '
        'in a proposed solution. Be specific and concise.\n\n'
        'Focus on:\n'
        '- Hidden or unjustified assumptions\n'
        '- Invalid mathematical steps\n'
        '- Arithmetic or algebraic errors\n'
        '- Overcounting / undercounting\n'
        '- Incorrect theorem application\n'
        '- Missing edge cases\n\n'
        'State the key issue in 1-2 sentences.'
    )

    tool_prompt = (
        'Use this tool to execute Python code for:\n'
        '- Complex calculations that would be error-prone by hand\n'
        '- Numerical verification of analytical results\n'
        '- Generating examples or testing conjectures\n'
        '- Brute-force verification for small cases\n\n'
        'The environment is a stateful Jupyter notebook with math, numpy, '
        'sympy, itertools, collections, and mpmath pre-imported.\n'
        'Always use print() to display results.\n\n'
        'Code should support your mathematical reasoning, not replace it.'
    )

    preference_prompt = (
        'You have access to `math`, `numpy`, and `sympy` for:\n\n'
        '# Symbolic Computation (sympy):\n'
        '- Algebraic manipulation and simplification\n'
        '- Solving equations and systems of equations\n'
        '- Number theory functions (primes, divisors, modular arithmetic)\n'
        '- Polynomial operations and factorization\n\n'
        '# Numerical Computation (numpy):\n'
        '- Array operations and linear algebra\n'
        '- Efficient numerical calculations\n\n'
        '# Mathematical Functions (math):\n'
        '- Standard mathematical functions\n'
        '- Constants like pi and e\n\n'
        'Best Practices:\n'
        '- Use sympy for exact symbolic answers when possible\n'
        '- Use numpy for numerical verification\n'
        '- Combine symbolic and numerical approaches\n'
        '- Validate results against known cases'
    )

    @staticmethod
    def build_judge_user_prompt(problem_text: str,
                                candidates: list[dict]) -> str:
        parts = [f'Problem:\n{problem_text}\n\n']
        parts.append(f'Below are {len(candidates)} candidate solutions. '
                     'Each provides a final answer and a brief rationale.\n\n')
        for i, c in enumerate(candidates, 1):
            parts.append(f'--- Candidate {i} ---\n')
            parts.append(f'Answer: {c["answer"]}\n')
            parts.append(f'Rationale: {c["rationale"]}\n\n')
        parts.append(
            'Analyze each candidate\'s reasoning for potential errors. '
            'Which answer is most likely correct? '
            'Provide your selection as \\boxed{answer}.'
        )
        return ''.join(parts)

    @staticmethod
    def build_critique_user_prompt(problem_text: str,
                                   candidate: dict) -> str:
        return (
            f'Problem:\n{problem_text}\n\n'
            f'A candidate solution proposes the answer {candidate["answer"]} '
            f'with the following reasoning:\n{candidate["rationale"]}\n\n'
            'Identify the most critical potential error or questionable '
            'assumption. State the key issue in 1-2 sentences.'
        )

    @staticmethod
    def build_rescue_user_input(problem_text: str, mistake_note: str,
                                preference_prompt: str) -> str:
        return (
            f'{problem_text}\n\n'
            f'IMPORTANT — A previous attempt identified this potential error '
            f'to avoid:\n"{mistake_note}"\n\n'
            f'Solve from scratch, taking care to avoid that error. '
            f'Consider alternative approaches.\n\n{preference_prompt}'
        )


set_seed(CFG.seed)


# ============================================================
# SECTION 5 — CHAT TEMPLATE
# ============================================================

class AIMO3Template:
    """Renders openai_harmony messages for both tool-enabled and tool-free
    prompts (judge / critique)."""

    @staticmethod
    def _system_content_with_tools(system_prompt: str,
                                   tool_config: ToolNamespaceConfig
                                   ) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    @staticmethod
    def _system_content_no_tools(system_prompt: str) -> SystemContent:
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
        )

    @classmethod
    def apply_chat_template(cls, system_prompt: str, user_prompt: str,
                            tool_config: ToolNamespaceConfig
                            ) -> list[Message]:
        sc = cls._system_content_with_tools(system_prompt, tool_config)
        return [
            Message.from_role_and_content(Role.SYSTEM, sc),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]

    @classmethod
    def apply_no_tool_template(cls, system_prompt: str,
                               user_prompt: str) -> list[Message]:
        sc = cls._system_content_no_tools(system_prompt)
        return [
            Message.from_role_and_content(Role.SYSTEM, sc),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]


# ============================================================
# SECTION 6 — JUPYTER SANDBOX
# ============================================================

class AIMO3Sandbox:
    """Persistent Jupyter kernel for code execution during tool-use turns."""

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout: float):
        self._default_timeout = timeout
        self._owns_kernel = False
        self._client = None
        self._km = None

        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
        env['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '0'
        env['JUPYTER_PLATFORM_DIRS'] = '1'
        env['PYTHONWARNINGS'] = 'ignore'
        env['MPLBACKEND'] = 'Agg'

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(
            env=env,
            extra_arguments=['--Application.log_level=CRITICAL'],
        )
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._default_timeout)
        self._owns_kernel = True

        self.execute(
            'import math\nimport numpy\nimport sympy\n'
            'import itertools\nimport collections\nimport mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def _format_error(self, traceback: list[str]) -> str:
        clean = []
        for frame in traceback:
            clean_frame = re.sub(r'\x1b\[[0-9;]*m', '', frame)
            if 'File "' in clean_frame and 'ipython-input' not in clean_frame:
                continue
            clean.append(clean_frame)
        return ''.join(clean)

    def execute(self, code: str, timeout: float | None = None) -> str:
        client = self._client
        effective_timeout = timeout or self._default_timeout
        msg_id = client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False,
        )
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        start = time.time()

        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return (f'[ERROR] Execution timed out after '
                        f'{effective_timeout} seconds')
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue
            if msg.get('parent_header', {}).get('msg_id') != msg_id:
                continue

            msg_type = msg.get('msg_type')
            content = msg.get('content', {})

            if msg_type == 'stream':
                text = content.get('text', '')
                (stdout_parts if content.get('name') == 'stdout'
                 else stderr_parts).append(text)
            elif msg_type == 'error':
                stderr_parts.append(
                    self._format_error(content.get('traceback', [])))
            elif msg_type in {'execute_result', 'display_data'}:
                text = content.get('data', {}).get('text/plain')
                if text:
                    stdout_parts.append(
                        text if text.endswith('\n') else f'{text}\n')
            elif msg_type == 'status':
                if content.get('execution_state') == 'idle':
                    break

        stdout = ''.join(stdout_parts)
        stderr = ''.join(stderr_parts)
        if stderr:
            return f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
        return stdout if stdout.strip() else (
            '[WARN] No output. Use print() to see results.')

    def reset(self):
        self.execute(
            '%reset -f\nimport math\nimport numpy\nimport sympy\n'
            'import itertools\nimport collections\nimport mpmath\n'
            'mpmath.mp.dps = 64\n'
        )

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    def __del__(self):
        self.close()


# ============================================================
# SECTION 7 — TOOL HANDLER
# ============================================================

class AIMO3Tool:
    """Wraps a sandbox to handle tool-call messages from the model."""

    def __init__(self, local_jupyter_timeout: float, tool_prompt: str,
                 sandbox=None):
        self._local_jupyter_timeout = local_jupyter_timeout
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._owns_session = sandbox is None
        self._execution_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def _ensure_session(self):
        if self._jupyter_session is None:
            with self._init_lock:
                if self._jupyter_session is None:
                    self._jupyter_session = AIMO3Sandbox(
                        timeout=self._local_jupyter_timeout)

    @staticmethod
    def _ensure_last_print(code: str) -> str:
        lines = code.strip().split('\n')
        if not lines:
            return code
        last = lines[-1].strip()
        if not last or 'print' in last or 'import' in last or last.startswith('#'):
            return code
        lines[-1] = 'print(' + last + ')'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name='python', description=self.instruction, tools=[])

    def _make_response(self, output: str, channel: str | None = None
                       ) -> Message:
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        msg = Message(author=author, content=[content]).with_recipient(
            'assistant')
        if channel:
            msg = msg.with_channel(channel)
        return msg

    def process_sync_plus(self, message: Message) -> list[Message]:
        self._ensure_session()
        raw = message.content[0].text
        final = self._ensure_last_print(raw)
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final)
            except TimeoutError as exc:
                output = f'[ERROR] {exc}'
        return [self._make_response(output, channel=message.channel)]


# ============================================================
# SECTION 8 — PROBLEM STATE  (thread-safe)
# ============================================================

class ProblemState:
    """Holds all mutable state for a single problem across concurrent
    attempts, judges, and critiques."""

    def __init__(self, problem_text: str, deadline: float, budget: float):
        self.problem_text = problem_text
        self.deadline = deadline
        self.budget = budget
        self._lock = threading.Lock()

        self.histogram: Counter = Counter()
        self.variant_map: dict[int, list[str]] = defaultdict(list)
        self.attempts: list[dict] = []
        self.candidates: list[dict] = []

        self.total_submitted = 0
        self.total_completed = 0

        self.genselect_votes: Counter = Counter()
        self.critique_notes: list[str] = []

        self.final_answer: Optional[int] = None
        self.finalized = False

    # --- mutators (called from worker threads) ---

    def add_attempt(self, result: dict) -> None:
        with self._lock:
            self.attempts.append(result)
            self.total_completed += 1
            answer = result.get('answer')
            if answer is not None:
                self.histogram[answer] += 1
                self.variant_map[answer].append(result['prompt_variant'])
                self.candidates.append({
                    'answer': answer,
                    'rationale': result.get('rationale_summary', ''),
                    'attempt_id': result['attempt_id'],
                    'entropy': result.get('entropy', float('inf')),
                })

    def increment_submitted(self) -> None:
        with self._lock:
            self.total_submitted += 1

    def finalize(self, answer: Optional[int] = None) -> None:
        with self._lock:
            if not self.finalized:
                self.finalized = True
                if answer is not None:
                    self.final_answer = answer

    def add_genselect_vote(self, answer: int) -> None:
        with self._lock:
            self.genselect_votes[answer] += 1

    def add_critique_note(self, note: str) -> None:
        with self._lock:
            self.critique_notes.append(note)

    # --- readers (snapshot for scheduler decisions) ---

    def snapshot(self) -> dict:
        with self._lock:
            return {
                'histogram': Counter(self.histogram),
                'variant_map': {k: list(v)
                                for k, v in self.variant_map.items()},
                'candidates': list(self.candidates),
                'total_completed': self.total_completed,
                'total_submitted': self.total_submitted,
                'genselect_votes': Counter(self.genselect_votes),
                'finalized': self.finalized,
            }


# ============================================================
# SECTION 9 — ADAPTIVE STOPPING
# ============================================================

class AdaptiveStopper:
    """Decides when we have enough evidence to finalize an answer."""

    @staticmethod
    def should_finalize(histogram: Counter,
                        variant_map: dict[int, list[str]],
                        cfg: type = CFG) -> bool:
        total = sum(histogram.values())
        if total < cfg.stop_tier1_count:
            return False

        top_answer, top_count = histogram.most_common(1)[0]
        share = top_count / total

        # Tier 1: very-early strong consensus
        if top_count >= cfg.stop_tier1_count and share >= cfg.stop_tier1_share:
            return True

        # Tier 2: standard consensus + variant diversity
        if top_count >= cfg.stop_tier2_count and share >= cfg.stop_tier2_share:
            variants = set(variant_map.get(top_answer, []))
            if len(variants) >= cfg.stop_tier2_variants:
                return True

        # Tier 3: overwhelming agreement regardless of share
        if top_count >= cfg.stop_tier3_count:
            return True

        return False


# ============================================================
# SECTION 10 — TEMPERATURE & PHASE SCHEDULER
# ============================================================

class PhaseScheduler:
    """Chooses prompt variant and temperature for the next attempt based on
    histogram concentration and attempt count."""

    @staticmethod
    def is_concentrated(histogram: Counter, threshold: float = 0.50) -> bool:
        if not histogram:
            return False
        total = sum(histogram.values())
        if total < 4:
            return False
        return histogram.most_common(1)[0][1] / total >= threshold

    @staticmethod
    def get_phase(total_completed: int, concentrated: bool) -> str:
        if total_completed < CFG.explore_batch:
            return 'explore'
        if concentrated:
            return 'exploit'
        if total_completed < 16:
            return 'explore'
        return 'exploit'

    @staticmethod
    def get_temperature(phase: str, concentrated: bool) -> float:
        if phase == 'explore':
            return CFG.explore_temp
        if phase == 'exploit':
            return CFG.exploit_temp if concentrated else CFG.exploit_temp_diffuse
        if phase == 'rescue':
            return CFG.rescue_temp
        return CFG.explore_temp

    @staticmethod
    def get_min_p(phase: str) -> float:
        return {
            'explore': CFG.explore_min_p,
            'exploit': CFG.exploit_min_p,
            'rescue': CFG.rescue_min_p,
        }.get(phase, CFG.explore_min_p)


# ============================================================
# SECTION 11 — TIME-BUDGET MANAGER
# ============================================================

class TimeBudgetManager:
    """Allocates per-problem time budgets so that easy problems yield
    savings to harder ones later in the notebook."""

    def __init__(self, notebook_start: float, cfg: type = CFG):
        self.notebook_start = notebook_start
        self.cfg = cfg
        self.solved = 0

    def get_budget(self) -> float:
        elapsed = time.time() - self.notebook_start
        remaining_time = self.cfg.notebook_limit - elapsed
        remaining_problems = max(1, self.cfg.total_problems - self.solved)
        reserved = (remaining_problems - 1) * self.cfg.min_budget
        budget = remaining_time - reserved
        return float(max(self.cfg.min_budget,
                         min(budget, self.cfg.max_budget)))

    def mark_solved(self) -> None:
        self.solved += 1


# ============================================================
# SECTION 12 — ANSWER SELECTOR
# ============================================================

class AnswerSelector:
    """Combines histogram votes, GenSelect-lite votes, and entropy-weighted
    scoring to pick the final answer."""

    @staticmethod
    def select(state: ProblemState) -> int:
        snap = state.snapshot()
        histogram = snap['histogram']
        genselect = snap['genselect_votes']

        if not histogram and not genselect:
            return 0

        combined: Counter = Counter()

        for answer, count in histogram.items():
            combined[answer] += count

        for answer, count in genselect.items():
            combined[answer] += count * 3

        # Entropy-weighted bonus from individual candidates
        for cand in snap['candidates']:
            answer = cand['answer']
            entropy = cand['entropy']
            combined[answer] += 1.0 / max(entropy, 1e-9) * 0.1

        if not combined:
            return 0
        return combined.most_common(1)[0][0]


# ============================================================
# SECTION 13 — MAIN SOLVER
# ============================================================

class AIMO3Solver:
    """End-to-end adaptive inference pipeline.

    Architecture:
      CPU-A (prompt builder) -> GPU-B (vLLM) -> CPU-C (result processor)
    all overlapped via concurrent futures so the GPU is never idle.
    """

    def __init__(self, cfg: type, port: int = 8000):
        self.cfg = cfg
        self.port = port
        self.base_url = f'http://0.0.0.0:{port}/v1'
        self.api_key = 'sk-local'

        self.template = AIMO3Template()
        self.prompts = PromptLibrary()
        self.encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = (
            self.encoding.stop_tokens_for_assistant_actions())

        self._preload_model_weights()
        self.server_process = self._start_server()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.cfg.session_timeout,
        )

        self._wait_for_server()
        self._initialize_kernels()

        self.notebook_start = time.time()
        self.budget_mgr = TimeBudgetManager(self.notebook_start, self.cfg)

    # ------ server lifecycle ------

    def _preload_model_weights(self) -> None:
        print(f'Loading model weights from {self.cfg.model_path} '
              f'into OS Page Cache...')
        t0 = time.time()
        files = []
        total_size = 0
        for root, _, fnames in os.walk(self.cfg.model_path):
            for fn in fnames:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    files.append(fp)
                    total_size += os.path.getsize(fp)

        def _read(path: str) -> None:
            with open(path, 'rb') as f:
                while f.read(1 << 30):
                    pass

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            list(ex.map(_read, files))
        print(f'Processed {len(files)} files '
              f'({total_size / 1e9:.2f} GB) in {time.time()-t0:.2f}s\n')

    def _start_server(self) -> subprocess.Popen:
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--seed', str(self.cfg.seed),
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', '1',
            '--max-num-seqs', str(self.cfg.batch_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--host', '0.0.0.0',
            '--port', str(self.port),
            '--dtype', self.cfg.dtype,
            '--kv-cache-dtype', self.cfg.kv_cache_dtype,
            '--max-model-len', str(self.cfg.context_tokens),
            '--stream-interval', str(self.cfg.stream_interval),
            '--async-scheduling',
            '--disable-log-stats',
            '--enable-prefix-caching',
        ]
        self.log_file = open('vllm_server.log', 'w')
        return subprocess.Popen(
            cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _wait_for_server(self) -> None:
        print('Waiting for vLLM server...')
        t0 = time.time()
        for _ in range(self.cfg.server_timeout):
            rc = self.server_process.poll()
            if rc is not None:
                self.log_file.flush()
                with open('vllm_server.log', 'r') as f:
                    logs = f.read()
                raise RuntimeError(
                    f'Server died with code {rc}. Logs:\n{logs}')
            try:
                self.client.models.list()
                print(f'Server ready ({time.time()-t0:.2f}s)\n')
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError('Server failed to start (timeout)')

    def _initialize_kernels(self) -> None:
        print(f'Initializing {self.cfg.workers} Jupyter kernels...')
        t0 = time.time()
        self.sandbox_pool: queue.Queue = queue.Queue()

        def _make():
            return AIMO3Sandbox(timeout=self.cfg.jupyter_timeout)

        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(_make) for _ in range(self.cfg.workers)]
            for f in as_completed(futs):
                self.sandbox_pool.put(f.result())
        print(f'Kernels ready ({time.time()-t0:.2f}s)\n')

    # ------ answer scanning ------

    @staticmethod
    def _scan_for_answer(text: str) -> Optional[int]:
        for pattern in [r'\\boxed\s*\{\s*([0-9,]+)\s*\}',
                        r'final\s+answer\s+is\s*([0-9,]+)']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    val = int(matches[-1].replace(',', ''))
                    if 0 <= val <= 99999:
                        return val
                except ValueError:
                    pass
        return None

    # ------ single solve attempt (multi-turn with tools) ------

    def _process_attempt(
        self,
        user_input: str,
        system_prompt: str,
        attempt_idx: int,
        stop_event: threading.Event,
        deadline: float,
        temperature: float,
        prompt_variant: str,
        min_p: float = 0.02,
    ) -> dict:
        result = {
            'attempt_id': attempt_idx,
            'prompt_variant': prompt_variant,
            'answer': None,
            'python_calls': 0,
            'python_errors': 0,
            'token_count': 0,
            'entropy': float('inf'),
            'rationale_summary': '',
            'elapsed': 0.0,
        }
        if stop_event.is_set() or time.time() > deadline:
            return result

        t0 = time.time()
        sandbox = None
        local_tool = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        final_answer = None
        logprobs_buf: list = []
        response_text_buf: list[str] = []

        attempt_seed = int(math.pow(self.cfg.seed + attempt_idx, 2))

        try:
            sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
            local_tool = AIMO3Tool(
                local_jupyter_timeout=self.cfg.jupyter_timeout,
                tool_prompt=self.prompts.tool_prompt,
                sandbox=sandbox,
            )
            messages = self.template.apply_chat_template(
                system_prompt, user_input, local_tool.tool_config)
            conversation = Conversation.from_messages(messages)

            for _ in range(self.cfg.turns):
                if stop_event.is_set() or time.time() > deadline:
                    break

                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT)
                max_tok = self.cfg.context_tokens - len(prompt_ids)
                if max_tok < self.cfg.buffer_tokens:
                    break

                stream = self.client.completions.create(
                    model=self.cfg.served_model_name,
                    temperature=temperature,
                    logprobs=self.cfg.top_logprobs,
                    max_tokens=max_tok,
                    prompt=prompt_ids,
                    seed=attempt_seed,
                    stream=True,
                    extra_body={
                        'min_p': min_p,
                        'stop_token_ids': self.stop_token_ids,
                        'return_token_ids': True,
                    },
                )

                try:
                    token_buf: list[int] = []
                    text_chunks: list[str] = []

                    for chunk in stream:
                        if stop_event.is_set() or time.time() > deadline:
                            break
                        new_tokens = chunk.choices[0].token_ids
                        new_text = chunk.choices[0].text or ''
                        if new_tokens:
                            token_buf.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                            response_text_buf.append(new_text)

                            clp = chunk.choices[0].logprobs
                            if clp and clp.top_logprobs:
                                logprobs_buf.extend(clp.top_logprobs)

                        if new_text and '}' in new_text:
                            window = ''.join(
                                text_chunks[-self.cfg.search_tokens:])
                            ans = self._scan_for_answer(window)
                            if ans is not None:
                                final_answer = ans
                                break
                finally:
                    stream.close()

                if final_answer is not None:
                    break
                if not token_buf:
                    break

                new_msgs = (
                    self.encoding.parse_messages_from_completion_tokens(
                        token_buf, Role.ASSISTANT))
                conversation.messages.extend(new_msgs)
                last_msg = new_msgs[-1]

                if last_msg.channel == 'final':
                    final_answer = self._scan_for_answer(
                        last_msg.content[0].text)
                    break

                if last_msg.recipient == 'python':
                    python_calls += 1
                    tool_resps = local_tool.process_sync_plus(last_msg)
                    resp_text = tool_resps[0].content[0].text
                    if (resp_text.startswith('[ERROR]')
                            or 'Traceback' in resp_text
                            or 'Error:' in resp_text):
                        python_errors += 1
                    conversation.messages.extend(tool_resps)

        except Exception:
            python_errors += 1
        finally:
            if sandbox is not None:
                sandbox.reset()
                self.sandbox_pool.put(sandbox)

        full_text = ''.join(response_text_buf)
        rationale = full_text[-1000:] if len(full_text) > 1000 else full_text

        mean_entropy = self._mean_entropy(logprobs_buf)

        result.update({
            'answer': final_answer,
            'python_calls': python_calls,
            'python_errors': python_errors,
            'token_count': total_tokens,
            'entropy': mean_entropy,
            'rationale_summary': rationale,
            'elapsed': time.time() - t0,
        })
        return result

    @staticmethod
    def _mean_entropy(logprobs_buf: list) -> float:
        if not logprobs_buf:
            return float('inf')
        total, count = 0.0, 0
        for top_lp in logprobs_buf:
            if not isinstance(top_lp, dict) or not top_lp:
                continue
            h = 0.0
            for lp in top_lp.values():
                p = math.exp(lp)
                if p > 0:
                    h -= p * math.log2(p)
            total += h
            count += 1
        return total / count if count else float('inf')

    # ------ GenSelect-lite judge round ------

    def _run_judge_round(self, problem_text: str,
                         candidates: list[dict],
                         seed: int, deadline: float) -> Optional[int]:
        if time.time() > deadline - 15:
            return None
        user_prompt = PromptLibrary.build_judge_user_prompt(
            problem_text, candidates)
        messages = self.template.apply_no_tool_template(
            self.prompts.judge_system, user_prompt)
        conversation = Conversation.from_messages(messages)
        prompt_ids = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT)
        max_tok = min(self.cfg.genselect_max_tokens,
                      self.cfg.context_tokens - len(prompt_ids))
        if max_tok < 64:
            return None
        try:
            resp = self.client.completions.create(
                model=self.cfg.served_model_name,
                prompt=prompt_ids,
                temperature=self.cfg.genselect_temp,
                max_tokens=max_tok,
                seed=seed,
                extra_body={'stop_token_ids': self.stop_token_ids},
            )
            return self._scan_for_answer(resp.choices[0].text)
        except Exception:
            return None

    # ------ critique call ------

    def _run_critique(self, problem_text: str, candidate: dict,
                      seed: int, deadline: float) -> Optional[str]:
        if time.time() > deadline - 15:
            return None
        user_prompt = PromptLibrary.build_critique_user_prompt(
            problem_text, candidate)
        messages = self.template.apply_no_tool_template(
            self.prompts.critique_system, user_prompt)
        conversation = Conversation.from_messages(messages)
        prompt_ids = self.encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT)
        max_tok = min(self.cfg.critique_max_tokens,
                      self.cfg.context_tokens - len(prompt_ids))
        if max_tok < 64:
            return None
        try:
            resp = self.client.completions.create(
                model=self.cfg.served_model_name,
                prompt=prompt_ids,
                temperature=self.cfg.critique_temp,
                max_tokens=max_tok,
                seed=seed,
                extra_body={'stop_token_ids': self.stop_token_ids},
            )
            note = resp.choices[0].text.strip()[:300]
            return note if note else None
        except Exception:
            return None

    # ------ helpers for the main loop ------

    @staticmethod
    def _deduplicated_candidates(candidates: list[dict],
                                 histogram: Counter,
                                 max_k: int = 12) -> list[dict]:
        by_answer: dict[int, list[dict]] = defaultdict(list)
        for c in candidates:
            by_answer[c['answer']].append(c)
        result = []
        for answer, _ in histogram.most_common(max_k):
            if answer in by_answer:
                best = min(by_answer[answer],
                           key=lambda x: x['entropy'])
                result.append(best)
        return result

    def _is_diffuse(self, histogram: Counter) -> bool:
        if not histogram:
            return True
        total = sum(histogram.values())
        top = histogram.most_common(1)[0][1]
        return top / total < self.cfg.critique_threshold_share

    # ------ main problem-solving pipeline ------

    def solve_problem(self, problem_text: str) -> int:
        print(f'\nProblem: {problem_text[:120]}...\n')

        budget = self.budget_mgr.get_budget()
        deadline = time.time() + budget
        state = ProblemState(problem_text, deadline, budget)
        user_input_base = f'{problem_text} {self.prompts.preference_prompt}'

        print(f'Budget: {budget:.0f}s  |  Remaining problems: '
              f'{self.cfg.total_problems - self.budget_mgr.solved}\n')

        stop_event = threading.Event()
        active: dict = {}               # future -> task_info
        next_idx = 0
        genselect_started = False
        critique_started = False
        rescue_submitted = 0

        extra_threads = 6               # headroom for judge / critique
        pool = ThreadPoolExecutor(
            max_workers=self.cfg.workers + extra_threads)

        try:
            # ---- Phase 1: initial explore batch ----
            initial = min(self.cfg.explore_batch, self.cfg.max_in_flight)
            for _ in range(initial):
                f = pool.submit(
                    self._process_attempt,
                    user_input_base, self.prompts.explore_system,
                    next_idx, stop_event, deadline,
                    self.cfg.explore_temp, 'explore',
                    self.cfg.explore_min_p,
                )
                active[f] = {'type': 'generate', 'idx': next_idx}
                state.increment_submitted()
                next_idx += 1

            # ---- reactive processing loop ----
            while active and not state.finalized and time.time() < deadline:
                done, _ = wait(active.keys(), timeout=0.5,
                               return_when=FIRST_COMPLETED)

                for fut in done:
                    info = active.pop(fut)
                    try:
                        result = fut.result()
                    except Exception:
                        continue

                    # --- dispatch by task type ---

                    if info['type'] == 'generate':
                        state.add_attempt(result)
                        snap = state.snapshot()

                        # adaptive stopping check
                        if AdaptiveStopper.should_finalize(
                                snap['histogram'], snap['variant_map']):
                            state.finalize()
                            break

                        concentrated = PhaseScheduler.is_concentrated(
                            snap['histogram'])

                        # waterfall: keep submitting if below caps
                        gen_in_flight = sum(
                            1 for v in active.values()
                            if v['type'] == 'generate')
                        if (snap['total_submitted'] < self.cfg.max_attempts
                                and gen_in_flight < self.cfg.max_in_flight
                                and time.time() < deadline - 30):
                            phase = PhaseScheduler.get_phase(
                                snap['total_completed'], concentrated)
                            temp = PhaseScheduler.get_temperature(
                                phase, concentrated)
                            sys_prompt = (
                                self.prompts.explore_system
                                if phase == 'explore'
                                else self.prompts.exploit_system)
                            mp = PhaseScheduler.get_min_p(phase)
                            f2 = pool.submit(
                                self._process_attempt,
                                user_input_base, sys_prompt,
                                next_idx, stop_event, deadline,
                                temp, phase, mp,
                            )
                            active[f2] = {
                                'type': 'generate', 'idx': next_idx}
                            state.increment_submitted()
                            next_idx += 1

                        # trigger GenSelect once we have enough candidates
                        if (not genselect_started
                                and snap['total_completed']
                                    >= self.cfg.explore_batch
                                and len(snap['candidates'])
                                    >= self.cfg.genselect_min_candidates
                                and not state.finalized
                                and time.time() < deadline - 60):
                            genselect_started = True
                            deduped = self._deduplicated_candidates(
                                snap['candidates'], snap['histogram'])
                            if len(deduped) >= 2:
                                for r in range(self.cfg.genselect_rounds):
                                    subset = random.sample(
                                        deduped,
                                        min(self.cfg.genselect_subset_size,
                                            len(deduped)))
                                    jf = pool.submit(
                                        self._run_judge_round,
                                        problem_text, subset,
                                        self.cfg.seed + 5000 + r, deadline,
                                    )
                                    active[jf] = {
                                        'type': 'judge', 'round': r}

                        # trigger critique if histogram is diffuse
                        if (not critique_started
                                and snap['total_completed']
                                    >= self.cfg.critique_min_attempts
                                and self._is_diffuse(snap['histogram'])
                                and not state.finalized
                                and time.time() < deadline - 90):
                            critique_started = True
                            deduped = self._deduplicated_candidates(
                                snap['candidates'], snap['histogram'])
                            if deduped:
                                cf = pool.submit(
                                    self._run_critique,
                                    problem_text, deduped[0],
                                    self.cfg.seed + 9000, deadline,
                                )
                                active[cf] = {'type': 'critique'}

                    elif info['type'] == 'judge':
                        selected = result
                        if selected is not None:
                            state.add_genselect_vote(selected)

                    elif info['type'] == 'critique':
                        note = result
                        if (note
                                and rescue_submitted
                                    < self.cfg.max_rescue_attempts
                                and time.time() < deadline - 60):
                            state.add_critique_note(note)
                            rescue_input = PromptLibrary.build_rescue_user_input(
                                problem_text, note,
                                self.prompts.preference_prompt)
                            rf = pool.submit(
                                self._process_attempt,
                                rescue_input,
                                self.prompts.exploit_system,
                                next_idx, stop_event, deadline,
                                self.cfg.rescue_temp, 'rescue',
                                self.cfg.rescue_min_p,
                            )
                            active[rf] = {
                                'type': 'generate', 'idx': next_idx}
                            state.increment_submitted()
                            next_idx += 1
                            rescue_submitted += 1

                    if state.finalized:
                        break

        finally:
            stop_event.set()
            for f in list(active):
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            self.budget_mgr.mark_solved()

        # ---- display results ----
        if state.attempts:
            rows = []
            for a in state.attempts:
                rows.append({
                    'Attempt': a['attempt_id'] + 1,
                    'Variant': a['prompt_variant'],
                    'Answer': a['answer'],
                    'Tokens': a['token_count'],
                    'PyCalls': a['python_calls'],
                    'PyErrs': a['python_errors'],
                    'Entropy': round(a['entropy'], 3),
                    'Time': round(a['elapsed'], 1),
                })
            df = pd.DataFrame(rows)
            df['Answer'] = df['Answer'].astype('Int64')
            display(df)

        # ---- final answer selection ----
        snap = state.snapshot()
        if not snap['histogram'] and not snap['genselect_votes']:
            print('\nResult: 0\n')
            return 0

        final = AnswerSelector.select(state)

        # display vote table
        vote_rows = []
        combined: Counter = Counter()
        for ans, cnt in snap['histogram'].items():
            combined[ans] += cnt
        for ans, cnt in snap['genselect_votes'].items():
            combined[ans] += cnt * 3
        for ans in combined:
            vote_rows.append({
                'Answer': ans,
                'Histogram': snap['histogram'].get(ans, 0),
                'GenSelect': snap['genselect_votes'].get(ans, 0),
                'Combined': round(combined[ans], 1),
            })
        vote_rows.sort(key=lambda x: x['Combined'], reverse=True)
        display(pd.DataFrame(vote_rows))

        print(f'\nFinal Answer: {final}\n')
        return final

    # ------ cleanup ------

    def __del__(self):
        if hasattr(self, 'server_process'):
            self.server_process.terminate()
            self.server_process.wait()
        if hasattr(self, 'log_file'):
            self.log_file.close()
        if hasattr(self, 'sandbox_pool'):
            while not self.sandbox_pool.empty():
                try:
                    self.sandbox_pool.get_nowait().close()
                except Exception:
                    pass


# ============================================================
# SECTION 14 — ENTRY POINT
# ============================================================

solver = AIMO3Solver(CFG)


def predict(id_: pl.DataFrame, question: pl.DataFrame,
            answer: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    id_val = id_.item(0)
    q_text = question.item(0)
    gc.disable()
    final = solver.solve_problem(q_text)
    gc.enable()
    gc.collect()
    return pl.DataFrame({'id': id_val, 'answer': final})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )
