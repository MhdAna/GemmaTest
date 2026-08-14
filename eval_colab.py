import argparse
import gc
import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base MedGemma vs LoRA adapter on OpenI subset")
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Path to saved LoRA adapter folder")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with columns: path, section_findings, section_impression")
    parser.add_argument("--images", type=Path, required=True, help="Root folder containing images referenced by CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_output"), help="Folder for results")
    parser.add_argument("--model-id", type=str, default="google/medgemma-4b-it", help="Base model id")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction from CSV")
    parser.add_argument("--min-val", type=int, default=30, help="Minimum validation size")
    parser.add_argument("--max-val", type=int, default=200, help="Maximum validation size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation length")
    parser.add_argument("--limit", type=int, default=None, help="Optional hard limit of eval rows")
    parser.add_argument("--use-4bit", action="store_true", help="Load models with 4-bit quantization")
    return parser.parse_args()


def make_report(df: pd.DataFrame) -> pd.Series:
    if "section_findings" in df.columns and "section_impression" in df.columns:
        return (df["section_findings"].fillna("") + "\n" + df["section_impression"].fillna("")).str.strip()
    if "findings" in df.columns and "impression" in df.columns:
        return (df["findings"].fillna("") + "\n" + df["impression"].fillna("")).str.strip()
    if "report" in df.columns:
        return df["report"].fillna("").str.strip()
    raise ValueError("CSV missing report columns")


def keyword_recall(predictions: list[str], references: list[str]) -> float:
    keywords = [
        "bronchitis",
        "bronchovascular",
        "peribronchial",
        "interstitial",
        "hilar",
        "atelectasis",
        "pneumonia",
        "effusion",
        "cardiomegaly",
        "edema",
    ]
    total = 0.0
    count = 0
    for pred, ref in zip(predictions, references):
        ref_l = ref.lower()
        pred_l = pred.lower()
        active = [k for k in keywords if k in ref_l]
        if not active:
            continue
        hit = sum(1 for k in active if k in pred_l)
        total += hit / len(active)
        count += 1
    return total / count if count else 0.0


def build_eval_df(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv).reset_index(drop=True)
    if "path" not in df.columns:
        raise ValueError("CSV must contain a 'path' column")

    df["_report"] = make_report(df)
    df = df[df["_report"].str.len() > 20].reset_index(drop=True)

    random.seed(args.seed)
    idx = list(range(len(df)))
    random.shuffle(idx)

    n_val = int(len(df) * args.val_fraction)
    n_val = max(args.min_val, n_val)
    n_val = min(args.max_val, n_val)
    n_val = min(n_val, len(df))

    val_df = df.iloc[idx[:n_val]].copy().reset_index(drop=True)
    if args.limit is not None:
        val_df = val_df.iloc[: args.limit].copy().reset_index(drop=True)

    return val_df


def load_model(model_id: str, use_4bit: bool) -> AutoModelForImageTextToText:
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="auto")


def generate_reports(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    eval_df: pd.DataFrame,
    image_root: Path,
    max_new_tokens: int,
) -> list[str]:
    prompt_text = (
        "You are an expert radiologist. Analyze this chest X-ray and write a structured radiology report "
        "with Findings and Impression sections."
    )

    preds: list[str] = []
    for i, row in eval_df.iterrows():
        img_path = image_root / str(row["path"])
        if not img_path.exists():
            preds.append("")
            continue

        img = Image.open(img_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
        ]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[[img]], return_tensors="pt").to("cuda")

        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        text = processor.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        preds.append(text)

        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{len(eval_df)}")

    return preds


def score_with_evaluate(predictions: list[str], references: list[str]) -> dict[str, float]:
    try:
        import evaluate
    except ImportError:
        return {}

    results: dict[str, float] = {}

    try:
        rouge = evaluate.load("rouge")
        out = rouge.compute(predictions=predictions, references=references)
        results["rougeL"] = float(out.get("rougeL", 0.0))
    except Exception as exc:
        print(f"ROUGE failed: {exc}")

    try:
        bert = evaluate.load("bertscore")
        out = bert.compute(predictions=predictions, references=references, lang="en")
        f1 = out.get("f1", [])
        if f1:
            results["bertscore_f1"] = float(sum(f1) / len(f1))
    except Exception as exc:
        print(f"BERTScore failed: {exc}")

    return results


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images folder not found: {args.images}")

    eval_df = build_eval_df(args)
    refs = eval_df["_report"].tolist()

    print(f"Eval rows: {len(eval_df)}")
    print(f"Model: {args.model_id}")
    print(f"Adapter: {args.adapter_dir}")

    processor = AutoProcessor.from_pretrained(args.model_id)

    print("\n[1/3] Generating BASE predictions")
    base_model = load_model(args.model_id, args.use_4bit).eval()
    base_preds = generate_reports(base_model, processor, eval_df, args.images, args.max_new_tokens)
    del base_model
    clear_cuda()

    print("\n[2/3] Generating ADAPTER predictions")
    ft_base = load_model(args.model_id, args.use_4bit).eval()
    ft_model = PeftModel.from_pretrained(ft_base, str(args.adapter_dir)).eval()
    ft_preds = generate_reports(ft_model, processor, eval_df, args.images, args.max_new_tokens)
    del ft_model
    del ft_base
    clear_cuda()

    print("\n[3/3] Scoring")
    base_scores = score_with_evaluate(base_preds, refs)
    ft_scores = score_with_evaluate(ft_preds, refs)

    base_kw = keyword_recall(base_preds, refs)
    ft_kw = keyword_recall(ft_preds, refs)

    summary = {
        "n_eval": len(eval_df),
        "base_keyword_recall": base_kw,
        "ft_keyword_recall": ft_kw,
        "delta_keyword_recall": ft_kw - base_kw,
    }

    for k, v in base_scores.items():
        summary[f"base_{k}"] = v
    for k, v in ft_scores.items():
        summary[f"ft_{k}"] = v

    if "rougeL" in base_scores and "rougeL" in ft_scores:
        summary["delta_rougeL"] = ft_scores["rougeL"] - base_scores["rougeL"]
    if "bertscore_f1" in base_scores and "bertscore_f1" in ft_scores:
        summary["delta_bertscore_f1"] = ft_scores["bertscore_f1"] - base_scores["bertscore_f1"]

    out_df = eval_df[["path"]].copy()
    out_df["reference"] = refs
    out_df["base_prediction"] = base_preds
    out_df["ft_prediction"] = ft_preds
    out_csv = args.output_dir / "per_case_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame([summary])
    summary_csv = args.output_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\nDone")
    print(summary_df.to_string(index=False))
    print(f"Per-case predictions: {out_csv}")
    print(f"Metrics summary: {summary_csv}")


if __name__ == "__main__":
    main()
