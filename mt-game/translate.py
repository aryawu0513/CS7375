"""Machine translation layer.

Backends (all HF hub models tagged `translation`):
  mock       - no model; tags the text so you can test the UI without a GPU
  nllb-600m / nllb-1.3b / nllb-3.3b - facebook/nllb-200 seq2seq models
  seed-x     - ByteDance-Seed/Seed-X-PPO-7B, a 7B causal LM that can also
               explain its translation ("explain it in detail" prompt)
  translategemma - google/translategemma-4b-it (gated: accept the Gemma
               license on HF first); no explanation mode

Model size is a neutral factor, not a quality condition. Deliberately bad
translations come only from inject.py.

translate() returns a dict:
  text    - the translation
  words   - [(word, prob)] per output word (min token prob within the word)
  alts    - other beam hypotheses, if num_alts > 0
"""
import hashlib, re, threading

# code -> (NLLB code, English name for Seed-X prompts)
LANGS = {
    "en": ("eng_Latn", "English"),
    "zh": ("zho_Hans", "Chinese"),
    "es": ("spa_Latn", "Spanish"),
    "fr": ("fra_Latn", "French"),
    "de": ("deu_Latn", "German"),
    "ja": ("jpn_Jpan", "Japanese"),
    "ko": ("kor_Hang", "Korean"),
}

MODELS = {
    "nllb-600m": "facebook/nllb-200-distilled-600M",
    "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
    "nllb-3.3b": "facebook/nllb-200-3.3B",
    "seed-x": "ByteDance-Seed/Seed-X-PPO-7B",
    "translategemma": "google/translategemma-4b-it",
}


def make_translator(name, device=None):
    if name == "mock":
        return MockTranslator()
    if name == "seed-x":
        return SeedXTranslator(MODELS[name], device)
    if name == "translategemma":
        return TranslateGemmaTranslator(MODELS[name], device)
    return NLLBTranslator(MODELS[name], device)


class MockTranslator:
    supports_cot = True

    def translate(self, text, src, tgt, num_alts=0):
        if src == tgt or not text.strip():
            return dict(text=text, words=[], alts=[])
        out = f"[{src}->{tgt}] {text}"
        # deterministic fake probabilities so the confidence UI can be tested
        words = [(w, 0.3 + 0.7 * (int(hashlib.md5(w.encode()).hexdigest(), 16) % 100) / 100)
                 for w in out.split()]
        return dict(text=out, words=words, alts=[f"{out} (alt {i + 1})" for i in range(num_alts)])

    def explain(self, text, src, tgt, translation, on_chunk=None):
        return f"[mock reasoning] '{text}' was translated word by word from {src} to {tgt}."


class _HFTranslator:
    """Shared generate + per-word confidence for seq2seq and causal models."""

    def _load(self, device):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.lock = threading.Lock()

    def _generate(self, enc, num_alts, **kw):
        n = max(1, num_alts + 1)
        with self.lock, self.torch.no_grad():
            out = self.model.generate(
                **enc, num_beams=max(4, n), num_return_sequences=n, max_new_tokens=512,
                output_scores=True, return_dict_in_generate=True, **kw)
            scores = self.model.compute_transition_scores(
                out.sequences, out.scores, out.beam_indices, normalize_logits=False)
        T = scores.shape[1]
        gen = out.sequences[:, -T:]
        texts = [self.tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        words = self._words(gen[0], scores[0].exp().tolist())
        return dict(text=texts[0], words=words, alts=[t for t in texts[1:] if t != texts[0]])

    def _words(self, ids, probs):
        special = set(self.tok.all_special_ids)
        words = []
        for i, p in zip(ids.tolist(), probs):
            if i in special:
                continue
            piece = self.tok.convert_ids_to_tokens(i)
            s = self.tok.convert_tokens_to_string([piece])
            # CJK has no word boundary marker, so each token is its own unit
            if piece.startswith("▁") or not words or re.search(_CJK, s):
                words.append([s.strip(), p])
            else:
                words[-1][0] += s
                words[-1][1] = min(words[-1][1], p)
        return [(w, round(p, 3)) for w, p in words if w]


class NLLBTranslator(_HFTranslator):
    supports_cot = False

    def __init__(self, model_name, device=None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._load(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=self.dtype).to(self.device).eval()

    def translate(self, text, src, tgt, num_alts=0):
        if src == tgt or not text.strip():
            return dict(text=text, words=[], alts=[])
        self.tok.src_lang = LANGS[src][0]
        enc = self.tok(text, return_tensors="pt").to(self.device)
        bos = self.tok.convert_tokens_to_ids(LANGS[tgt][0])
        return self._generate(enc, num_alts, forced_bos_token_id=bos)


class SeedXTranslator(_HFTranslator):
    """Prompt format from the Seed-X model card; it has no chat template."""
    supports_cot = True

    def __init__(self, model_name, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._load(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype).to(self.device).eval()

    def _prompt(self, text, src, tgt, explain=False):
        verb = " and explain it in detail" if explain else ""
        return (f"Translate the following {LANGS[src][1]} sentence into {LANGS[tgt][1]}{verb}:\n"
                f"{text} <{tgt}>")

    def translate(self, text, src, tgt, num_alts=0):
        if src == tgt or not text.strip():
            return dict(text=text, words=[], alts=[])
        enc = self.tok(self._prompt(text, src, tgt), return_tensors="pt").to(self.device)
        return self._generate(enc, num_alts)

    # Without an opener the rationale comes out in English; this one makes it
    # Chinese (the receiver's language). Other targets are untested.
    COT_OPENERS = {"zh": " 好的，"}

    def explain(self, text, src, tgt, translation, on_chunk=None, max_new_tokens=400, chunk=48):
        """Rationale for a given translation. The explain prompt alone mostly
        returns just the translation, so we prefill its output format
        ("<translation>\n\n[COT]") with the translation actually shown.

        Generated in chunks, releasing the model lock in between so a new
        message's translation never waits behind a whole rationale;
        on_chunk(text_so_far) is called after each chunk."""
        opener = self.COT_OPENERS.get(tgt, "")
        prompt = self._prompt(text, src, tgt, explain=True) + translation + "\n\n[COT]" + opener
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        start, done = ids.shape[1], False
        while not done and ids.shape[1] - start < max_new_tokens:
            with self.lock, self.torch.no_grad():
                out = self.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                          max_new_tokens=chunk, do_sample=False)
            done = out.shape[1] - ids.shape[1] < chunk or out[0, -1].item() == self.tok.eos_token_id
            ids = out
            if on_chunk:
                on_chunk(opener.strip() + self.tok.decode(ids[0, start:], skip_special_tokens=True).strip())
        cut = "" if done else "…"
        return opener.strip() + self.tok.decode(ids[0, start:], skip_special_tokens=True).strip() + cut


class TranslateGemmaTranslator(_HFTranslator):
    """Uses the model's chat template, which builds the translation prompt from
    source/target language codes (ISO 639-1, plus script for Chinese)."""
    supports_cot = False
    CODES = {"zh": "zh-Hans"}

    def __init__(self, model_name, device=None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._load(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=self.dtype).to(self.device).eval()

    def translate(self, text, src, tgt, num_alts=0):
        if src == tgt or not text.strip():
            return dict(text=text, words=[], alts=[])
        msg = [{"role": "user", "content": [{
            "type": "text", "text": text,
            "source_lang_code": self.CODES.get(src, src),
            "target_lang_code": self.CODES.get(tgt, tgt)}]}]
        enc = self.processor.apply_chat_template(
            msg, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to(self.device)
        return self._generate(enc, num_alts)


_CJK = r"[぀-ヿ㐀-鿿豈-﫿]"


def tokens(s):
    """Words for space-delimited scripts, single characters for CJK."""
    return re.findall(rf"{_CJK}|[^\W\d_]+|\d+", s.lower())


def is_cjk(tok):
    return bool(re.fullmatch(_CJK, tok))
