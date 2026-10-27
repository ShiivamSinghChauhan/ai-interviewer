"""How the speech recognizer mangles AI jargon, and how to un-mangle it.

Nemotron has never heard of QLoRA, so "q laura" gets fixed here. The patterns come from running Indian, US and
UK voices through it. Order matters in a few places: QLoRA before LoRA, and "l l m" before "LLM as a judge".
"""
import re

_FIXES = [
    # models and libraries
    (r"\b(llama|llma|lama|llm)\s?(dot\s?)?c\s?p\s?p\b|\bllm\s?cpe\b", "llama.cpp"),
    (r"\bl?lama\s?index\b", "LlamaIndex"),
    (r"\blama\b", "Llama"),
    (r"\bmixt(r?[ae]ll?|ural|rall?)\b", "Mixtral"),
    (r"\b(qwen|quen|kwen|quin)\b", "Qwen"),
    (r"\bdeep\s?seek\b", "DeepSeek"),
    (r"\bopen\s?a\.?\s?i\b", "OpenAI"),
    (r"\bchat\s?g[\s.-]?p[\s.-]?t\b", "ChatGPT"),
    (r"\blang\s?chain\b", "LangChain"),
    (r"\blan[gd]?\s?gra(ph|f)\b", "LangGraph"),
    (r"\bhugging\s?face\b", "Hugging Face"),
    (r"\b(pie|py)\s?torch\b", "PyTorch"),
    (r"\bonyx\b", "ONNX"),
    (r"\bfast\s?api\b", "FastAPI"),
    (r"\bcube\s?ernetes\b|\bkubernetes\b", "Kubernetes"),
    (r"\bx\s?g\s?boost\b", "XGBoost"),
    (r"\bweights and bs\b", "Weights and Biases"),

    # LLMs and serving
    (r"\bv[\s.-]?l[\s.-]?l[\s.-]?m\b", "vLLM"),
    (r"\bl[\s.-]?l[\s.-]?m(s?)\b", r"LLM\1"),
    (r"\bllm[\s-]?(as[\s-]?a[\s-]?judge|ass?[\s-]?edge|assay[\s-]?judge)\b", "LLM-as-a-judge"),
    (r"\bpaged?\s?detention\b|\bpatient attention\b|\bpaste attention\b", "paged attention"),
    (r"\bk[\s.-]?v[\s.-]?(cache|cash)\b", "KV cache"),
    (r"\btop[\s-]?(p|pea|pie|pee)\b|\btoppy\b", "top-p"),
    (r"\btop[\s-]?(k|kay)\b", "top-k"),
    (r"\bm[\s.-]?o[\s.-]?e\b", "MoE"),
    (r"\bsoft\s?max\b", "softmax"),
    (r"\bm[\s.-]?c[\s.-]?p\b", "MCP"),
    (r"\ba genetic\b(?=\s+(ai|workflow|workflows|system|systems|rag|framework|pattern|patterns|loop))", "agentic"),

    # fine-tuning and alignment
    (r"\b(q|c|cu|kyu|ch)[\s-]?(laura|lora|lo\s?ra|l\.?o\.?r\.?a)\b|\bculo\s?ra\b", "QLoRA"),
    (r"\b(laura|lora|l\.o\.r\.a\.?)\b", "LoRA"),
    (r"\bp\s?f\s?d\b|\bp[\s.-]?e[\s.-]?f[\s.-]?t\b", "PEFT"),
    (r"\bfine\s(tun(e|ed|es|ing))\b", r"fine-\1"),
    (r"\br[\s.-]?l[\s.-]?h[\s.-]?f\b", "RLHF"),
    (r"\bd[\s.-]?p[\s.-]?o\b", "DPO"),
    (r"\bg[\s.-]?r[\s.-]?p[\s.-]?o\b", "GRPO"),
    (r"\b(evil|avail)\s+set\b", "eval set"),
    (r"\bml\s?ops\b", "MLOps"),

    # quantization and hardware
    (r"\bquanti[sz]ation\b|\bquantitization\b", "quantization"),
    (r"\bint[\s-]?(four|4)\b", "INT4"),
    (r"\bint[\s-]?(eight|8)\b", "INT8"),
    (r"\bb\s?float\s?16\b|\bb[\s.-]?f[\s.-]?16\b", "bf16"),
    (r"\bf[\s.-]?p[\s.-]?(16|8|32)\b", r"FP\1"),
    (r"\bg[\s.-]?p[\s.-]?t[\s.-]?q\b", "GPTQ"),
    (r"\ba[\s.-]?w[\s.-]?q\b", "AWQ"),
    (r"\bg[\s.-]?g[\s.-]?(u[\s.-]?)?f\b", "GGUF"),
    (r"\b(cuda|kuda)\b", "CUDA"),
    (r"\b([gct]pu|api)(s?)\b", lambda m: m.group(1).upper() + m.group(2)),

    # retrieval
    (r"\bragas\b", "RAGAS"),
    (r"\brag\b", "RAG"),
    (r"\b(rack|rg)\b(?=[\s-]+(pipeline|based|system|setup|application|app|architecture|chatbot))", "RAG"),
    (r"\bre[\s-]?[rl]ank(er|ers|ing|ed)\b", r"rerank\1"),
    (r"\bcross\s?encoder", "cross-encoder"),
    (r"\bb[\s.-]?[mn][\s.-]?(twenty[\s-]?five|25)\b", "BM25"),
    (r"\bt[\s.-]?f[\s.-]?i[\s.-]?d[\s.-]?f\b", "TF-IDF"),
    (r"\bcol\s?bert\b", "ColBERT"),
    (r"\bh[\s.-]?n[\s.-]?s[\s.-]?w\b", "HNSW"),
    (r"\b(?:em|im)\s?bedd?ings?\b", lambda m: "embeddings" if m.group(0).lower().endswith("s") else "embedding"),
    (r"\bq\s?(d|g)?rant?\b|\bqdran\b|\bq\s?(tron|ront)\b", "Qdrant"),
    (r"\bpine\s?cone\b", "Pinecone"),
    (r"\bfaiss\b", "FAISS"),
    (r"\bchroma\s?d\.?\s?b\b", "ChromaDB"),
    (r"\b(p\.?\s?g\.?|page)\s?vector\b", "pgvector"),
]
_FIXES = [(re.compile(pattern, re.IGNORECASE), fix) for pattern, fix in _FIXES]

_ROPE_CONTEXT = re.compile(r"rotary|positional|embedding|context (length|window)|extrapolat", re.IGNORECASE)


def fix_transcript(text: str) -> str:
    for pattern, fix in _FIXES:
        text = pattern.sub(fix, text)
    if _ROPE_CONTEXT.search(text):  # "rope" is only RoPE when they're talking about positions
        text = re.sub(r"\brope\b", "RoPE", text, flags=re.IGNORECASE)
    return text
