from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_SYSTEM_PROMPT = (
    "你是一位严谨、耐心的中文助理，擅长知识问答、逻辑推理与数学计算。"
    "请遵循用户要求输出；遇到不确定的事实请说明不确定，不要编造。"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _count_jsonl_lines(path: str, *, max_lines: Optional[int] = None) -> int:
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_lines is not None and n >= max_lines:
                break
    return n


def _load_existing_hashes(path: str, *, max_lines: int) -> set[str]:
    hashes: set[str] = set()
    if not os.path.isfile(path) or max_lines <= 0:
        return hashes
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta") if isinstance(obj, dict) else None
            if isinstance(meta, dict) and isinstance(meta.get("prompt_hash"), str):
                hashes.add(str(meta["prompt_hash"]))
    return hashes


def ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    seed: int,
    think: bool,
    temperature: float,
    top_p: float,
    timeout_s: float,
    session: Optional[requests.Session] = None,
) -> Tuple[str, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": bool(think),
        "options": {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "num_predict": int(max_new_tokens),
            "seed": int(seed),
        },
    }
    req = session or requests
    resp = req.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    text = content if isinstance(content, str) and content.strip() else ""
    if not text:
        text = data.get("response", "") or ""
    return str(text).strip(), data


def _pick_weighted(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(max(0.0, float(w)) for _, w in items)
    if total <= 0:
        return items[0][0]
    r = rng.random() * total
    acc = 0.0
    for name, w in items:
        acc += max(0.0, float(w))
        if r <= acc:
            return name
    return items[-1][0]


@dataclass(frozen=True)
class Prompt:
    idx: int
    category: str
    system: str
    user: str
    meta: Dict[str, Any]


def _math_arithmetic(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # Mix of integer arithmetic with parentheses.
    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    c = rng.randint(2, 99)
    d = rng.randint(2, 99)
    op1 = rng.choice(["+", "-"])
    op2 = rng.choice(["*", "//"])
    expr = f"({a} {op1} {b}) {op2} {c} + {d}"
    ans = (a + b if op1 == "+" else a - b)
    if op2 == "*":
        ans = ans * c
    else:
        ans = ans // c
    ans = ans + d
    prompt = (
        f"计算：{expr}\n\n"
        "要求：\n"
        "1) 给出最终结果（整数）。\n"
        "2) 简要写出关键步骤（不超过4步）。\n"
        "输出格式：\n"
        "步骤：...\n"
        "答案：..."
    )
    return prompt, {"ground_truth_int": int(ans), "expr": expr}


def _math_number_theory(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    import math

    a = rng.randint(10, 999)
    b = rng.randint(10, 999)
    g = math.gcd(a, b)
    l = a // g * b
    prompt = (
        f"求 {a} 和 {b} 的最大公约数与最小公倍数。\n\n"
        "要求：\n"
        "1) 输出两行：第一行 GCD=...；第二行 LCM=...\n"
        "2) 给出不超过3步的简要说明。"
    )
    return prompt, {"ground_truth_gcd": int(g), "ground_truth_lcm": int(l), "a": a, "b": b}


def _math_equation(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # Solve: ax + b = c
    a = rng.randint(2, 15)
    x = rng.randint(-30, 30)
    b = rng.randint(-50, 50)
    c = a * x + b
    prompt = (
        f"解一元一次方程：{a}x + ({b}) = {c}\n\n"
        "要求：给出 x 的值，并用不超过3步解释推导过程。输出以“x=”开头。"
    )
    return prompt, {"ground_truth_x": int(x), "a": a, "b": b, "c": c}


def _math_word_problem(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # Simple multi-step word problem (percentage + total).
    total = rng.randint(80, 500)
    sold_pct = rng.choice([20, 25, 30, 40, 45, 50, 60])
    rest = total * (100 - sold_pct) // 100
    prompt = (
        f"一家店进了 {total} 件商品，第一天卖出了 {sold_pct}% 。第二天又卖出了剩余商品的 1/2。\n"
        "问：第二天卖出了多少件？还剩多少件？\n\n"
        "要求：给出“第二天卖出=...，剩余=...”并简要说明计算步骤。"
    )
    day2 = rest // 2
    left = rest - day2
    return prompt, {"ground_truth_day2": int(day2), "ground_truth_left": int(left), "total": total, "sold_pct": sold_pct}


def _logic_syllogism(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    a = rng.choice(["鲸鱼", "海豚", "蝙蝠", "鸵鸟"])
    b = rng.choice(["哺乳动物", "鸟类", "鱼类"])
    c = rng.choice(["需要呼吸空气", "会下蛋", "生活在水中"])
    truth = rng.choice([True, False])
    if truth:
        prompt = (
            f"逻辑推理题：\n"
            f"前提1：所有 {a} 都是 {b}。\n"
            f"前提2：所有 {b} 都 {c}。\n"
            f"结论：所有 {a} 都 {c}。\n\n"
            "问：结论是否必然成立？请回答“成立/不成立”，并用不超过3句解释。"
        )
    else:
        prompt = (
            f"逻辑推理题：\n"
            f"前提1：所有 {a} 都是 {b}。\n"
            f"前提2：有些 {b} 不 {c}。\n"
            f"结论：所有 {a} 都 {c}。\n\n"
            "问：结论是否必然成立？请回答“成立/不成立”，并用不超过3句解释。"
        )
    # Not strictly ground-truth without formal semantics; keep meta only.
    return prompt, {"variant": "syllogism", "truth_case": bool(truth)}


def _logic_constraints(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # Simple constraint satisfaction: ordering with conditions.
    names = ["甲", "乙", "丙", "丁"]
    rng.shuffle(names)
    a, b, c, d = names
    prompt = (
        f"逻辑排序题：四个人 {a}、{b}、{c}、{d} 参加比赛，名次从第1到第4互不相同。\n"
        f"已知：\n"
        f"1) {a} 的名次比 {b} 靠前。\n"
        f"2) {c} 不是第1名也不是第4名。\n"
        f"3) {d} 的名次比 {c} 靠后。\n\n"
        "问：可能的名次安排有哪些？请列出所有可能（用“第1=..., 第2=..., 第3=..., 第4=...”格式，每种一行）。"
    )
    return prompt, {"variant": "constraints", "people": [a, b, c, d]}


def _reasoning_multi_hop(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    # Multi-hop quantitative reasoning.
    apples = rng.randint(8, 30)
    give = rng.randint(2, min(10, apples - 1))
    buy = rng.randint(3, 15)
    lose = rng.randint(1, min(8, apples + buy - give - 1))
    final = apples - give + buy - lose
    prompt = (
        f"推理题：小李原来有 {apples} 个苹果，先送给朋友 {give} 个，后来又买了 {buy} 个，回家路上丢了 {lose} 个。\n"
        "问：他最后有多少个苹果？\n\n"
        "要求：先给出结论，再给出不超过3步的简要推理。输出格式：\n"
        "结论：...\n"
        "推理：..."
    )
    return prompt, {"ground_truth_int": int(final), "variant": "multi_hop"}


_KNOWLEDGE_TOPICS = [
    ("knowledge_science", "什么是光合作用？它对生态系统有什么意义？"),
    ("knowledge_science", "解释一下“熵”的直观含义，并举一个生活中的例子。"),
    ("knowledge_history", "简述文艺复兴的主要特征与影响（不超过150字）。"),
    ("knowledge_history", "为什么工业革命会首先发生在英国？列出3个关键原因。"),
    ("knowledge_geo", "解释季风的形成原因，并说明它对东亚气候的影响。"),
    ("knowledge_cs", "解释什么是哈希表（Hash Table），以及它常见的冲突解决方式。"),
    ("knowledge_cs", "简述 HTTP 与 HTTPS 的区别，以及 HTTPS 主要解决了什么问题。"),
    ("knowledge_math", "用通俗语言解释什么是“导数”，并给出一个简单的应用场景。"),
]


def _knowledge_qa(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    cat, q = rng.choice(_KNOWLEDGE_TOPICS)
    prompt = (
        f"{q}\n\n"
        "要求：\n"
        "1) 用中文回答，结构清晰。\n"
        "2) 如果涉及专业术语，给出一句话解释。\n"
        "3) 不要胡编来源或数据；不确定就说明不确定。"
    )
    return prompt, {"topic_category": cat}


def build_prompt(*, idx: int, seed: int, system: str, weights: Dict[str, float]) -> Prompt:
    rng = random.Random(int(seed) + int(idx) * 1000003)
    category = _pick_weighted(
        rng,
        [
            ("math", float(weights.get("math", 1.0))),
            ("logic", float(weights.get("logic", 1.0))),
            ("reasoning", float(weights.get("reasoning", 1.0))),
            ("knowledge", float(weights.get("knowledge", 1.0))),
        ],
    )

    meta: Dict[str, Any] = {"category": category}
    if category == "math":
        kind = _pick_weighted(rng, [("arith", 0.35), ("nt", 0.25), ("eq", 0.20), ("word", 0.20)])
        if kind == "arith":
            user, m = _math_arithmetic(rng)
        elif kind == "nt":
            user, m = _math_number_theory(rng)
        elif kind == "eq":
            user, m = _math_equation(rng)
        else:
            user, m = _math_word_problem(rng)
        meta.update({"subtype": f"math_{kind}", **m})
    elif category == "logic":
        kind = _pick_weighted(rng, [("syllogism", 0.5), ("constraints", 0.5)])
        if kind == "syllogism":
            user, m = _logic_syllogism(rng)
        else:
            user, m = _logic_constraints(rng)
        meta.update({"subtype": f"logic_{kind}", **m})
    elif category == "reasoning":
        user, m = _reasoning_multi_hop(rng)
        meta.update({"subtype": "reasoning_multi_hop", **m})
    else:
        user, m = _knowledge_qa(rng)
        meta.update({"subtype": "knowledge_qa", **m})

    return Prompt(idx=idx, category=category, system=system, user=user, meta=meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic SFT JSONL by distilling a local Ollama model (e.g. qwen3:0.6b)."
    )
    parser.add_argument("--out_jsonl", type=str, required=True, help="Output JSONL path (appends if exists).")
    parser.add_argument("--ollama_url", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama_model", type=str, default="qwen3:0.6b")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument(
        "--target_total_samples",
        type=int,
        default=0,
        help="Stop when output file reaches this many lines (0 = run forever).",
    )
    parser.add_argument("--cold_min_samples", type=int, default=0, help="Exit early once at least N new samples were written.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout_s", type=float, default=600.0)
    parser.add_argument(
        "--health_check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast if Ollama is unreachable (default: enabled).",
    )
    parser.add_argument("--health_timeout_s", type=float, default=5.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_backoff_s", type=float, default=1.0)
    parser.add_argument(
        "--max_failures",
        type=int,
        default=200,
        help="Stop after this many failed prompts (0 = unlimited). Helps avoid hanging when Ollama is down.",
    )
    parser.add_argument("--min_answer_chars", type=int, default=4)
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Control Qwen3 thinking mode (default: disabled; distill concise steps via prompt).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--dedup_existing_lines", type=int, default=200000, help="How many existing lines to scan for prompt_hash dedup (0 to disable).")

    parser.add_argument("--w_math", type=float, default=0.38)
    parser.add_argument("--w_logic", type=float, default=0.22)
    parser.add_argument("--w_reasoning", type=float, default=0.25)
    parser.add_argument("--w_knowledge", type=float, default=0.15)

    args = parser.parse_args()

    out_path = str(args.out_jsonl)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if bool(args.health_check):
        try:
            url = str(args.ollama_url).rstrip("/") + "/api/tags"
            resp = requests.get(url, timeout=float(args.health_timeout_s))
            resp.raise_for_status()
            data = resp.json()
            names: List[str] = []
            for m in (data.get("models") or []):
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    names.append(str(m["name"]))
            if names and str(args.ollama_model) not in set(names):
                print(f"[warn] ollama model not found in tags: {args.ollama_model} (have {len(names)} models)", flush=True)
        except Exception as e:
            raise RuntimeError(
                f"Ollama health check failed for {args.ollama_url}. "
                f"Make sure `ollama serve` is running. Error: {e}"
            ) from e

    existing = _count_jsonl_lines(out_path)
    prompt_hashes = _load_existing_hashes(out_path, max_lines=int(args.dedup_existing_lines))
    if existing > 0:
        print(f"[data] existing={existing} lines | dedup_hashes={len(prompt_hashes)} (scan={args.dedup_existing_lines})")
    else:
        print("[data] existing=0 lines")

    target_total = int(args.target_total_samples)
    cold_min = int(args.cold_min_samples)
    if cold_min < 0:
        raise ValueError("--cold_min_samples must be >= 0")
    if target_total < 0:
        raise ValueError("--target_total_samples must be >= 0")

    # We only guarantee "at least N new samples" in cold mode; total target is optional.
    start_time = time.time()
    lock = threading.Lock()
    stop_event = threading.Event()
    written_new = 0
    written_total = existing
    next_idx = existing
    failed_prompts = 0

    weights = {
        "math": float(args.w_math),
        "logic": float(args.w_logic),
        "reasoning": float(args.w_reasoning),
        "knowledge": float(args.w_knowledge),
    }

    def should_stop_locked() -> bool:
        if stop_event.is_set():
            return True
        if cold_min > 0 and written_new >= cold_min:
            return True
        if target_total > 0 and written_total >= target_total:
            return True
        if int(args.max_failures) > 0 and failed_prompts >= int(args.max_failures):
            return True
        return False

    def alloc_prompt() -> Optional[Prompt]:
        nonlocal next_idx
        with lock:
            if should_stop_locked():
                return None
            idx = next_idx
            next_idx += 1
        return build_prompt(idx=idx, seed=int(args.seed), system=str(args.system), weights=weights)

    write_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(64, int(args.num_workers) * 4))

    def writer_thread() -> None:
        nonlocal written_new, written_total
        with open(out_path, "a", encoding="utf-8") as f:
            while True:
                try:
                    item = write_q.get(timeout=0.25)
                except queue.Empty:
                    with lock:
                        done = should_stop_locked()
                    if (stop_event.is_set() or done) and write_q.empty():
                        break
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                with lock:
                    written_new += 1
                    written_total += 1
                    n_new = written_new
                    n_total = written_total
                if int(args.log_every) > 0 and n_new % int(args.log_every) == 0:
                    dt = time.time() - start_time
                    rate = n_new / max(dt, 1e-6)
                    msg = f"[gen] new={n_new} total={n_total}"
                    if target_total > 0:
                        msg += f" target_total={target_total}"
                    msg += f" | {rate:.2f} samples/s"
                    print(msg, flush=True)
                write_q.task_done()

    def worker_thread(worker_id: int) -> None:
        nonlocal failed_prompts
        session = requests.Session()
        while not stop_event.is_set():
            prompt = alloc_prompt()
            if prompt is None:
                break

            prompt_hash = _sha1(prompt.system + "\n\n" + prompt.user)
            with lock:
                if prompt_hash in prompt_hashes:
                    continue
                prompt_hashes.add(prompt_hash)

            messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]

            last_err: Optional[str] = None
            for attempt in range(int(args.max_retries) + 1):
                if stop_event.is_set():
                    break
                try:
                    # Use a per-request seed for diversity while still being reproducible.
                    req_seed = int(args.seed) + int(prompt.idx) + int(worker_id) * 10007
                    text, meta = ollama_chat(
                        base_url=str(args.ollama_url),
                        model=str(args.ollama_model),
                        messages=messages,
                        max_new_tokens=int(args.max_new_tokens),
                        seed=req_seed,
                        think=bool(args.think),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        timeout_s=float(args.timeout_s),
                        session=session,
                    )
                    if len(text.strip()) < int(args.min_answer_chars):
                        raise RuntimeError("empty/too-short response")
                    record = {
                        "conversations": [
                            {"role": "system", "content": prompt.system},
                            {"role": "user", "content": prompt.user},
                            {"role": "assistant", "content": text.strip()},
                        ],
                        "meta": {
                            **prompt.meta,
                            "id": f"{prompt.category}_{prompt.idx:08d}",
                            "created_at": _utc_now_iso(),
                            "teacher": {"provider": "ollama", "model": str(args.ollama_model), "url": str(args.ollama_url)},
                            "prompt_hash": prompt_hash,
                            "ollama_meta": meta,
                        },
                    }
                    if stop_event.is_set():
                        last_err = "stopped"
                        break
                    try:
                        write_q.put(record, timeout=0.5)
                    except queue.Full:
                        last_err = "writer queue full"
                        continue
                    last_err = None
                    break
                except Exception as e:
                    last_err = str(e)
                    backoff = float(args.retry_backoff_s) * (2**attempt)
                    time.sleep(min(backoff, 30.0))

            if last_err is not None:
                with lock:
                    failed_prompts += 1
                    failures = failed_prompts
                    max_fail = int(args.max_failures)
                    if max_fail > 0 and failures >= max_fail:
                        stop_event.set()
                print(f"[warn] worker={worker_id} idx={prompt.idx} failed: {last_err}", flush=True)

    writer = threading.Thread(target=writer_thread, name="writer", daemon=True)
    writer.start()

    workers = []
    for wid in range(int(args.num_workers)):
        t = threading.Thread(target=worker_thread, args=(wid,), name=f"worker_{wid}", daemon=True)
        t.start()
        workers.append(t)

    try:
        while True:
            with lock:
                done = should_stop_locked()
                n_new = written_new
                n_total = written_total
            if done:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[gen] interrupted, stopping...", flush=True)
    finally:
        stop_event.set()

    for t in workers:
        t.join(timeout=2.0)
    try:
        write_q.join()
    except KeyboardInterrupt:
        pass
    writer.join(timeout=2.0)

    dt = time.time() - start_time
    print(f"[done] wrote_new={written_new} total={written_total} elapsed_s={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
