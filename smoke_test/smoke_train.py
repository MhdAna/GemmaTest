import os
import sys
import argparse
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from dicom_utils import load_dicom, dicom_to_pil

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

parser = argparse.ArgumentParser()
parser.add_argument("--synthetic", action="store_true",
                    help="Use existing dataset/ DICOMs instead of CheXpert Plus")
parser.add_argument("--openi", action="store_true",
                    help="Use OpenI dataset (PNGs) instead of CheXpert Plus DICOMs")
parser.add_argument("--images", type=Path, default=None,
                    help="Override image root folder (default: openi/ or chexpert_plus/)")
parser.add_argument("--batch-size", type=int, default=1,
                    help="Batch size (default: 1 for macOS stability)")
parser.add_argument("--max-seq-len", type=int, default=1024,
                    help="Processor max sequence length")
parser.add_argument("--single-view", action="store_true",
                    help="Use frontal image only even if lateral exists")
parser.add_argument("--mps-autocast", action="store_true",
                    help="Enable bfloat16 autocast on MPS (off by default for stability)")
parser.add_argument("--load-in-4bit", action="store_true",
                    help="Load model in 4-bit quantization (QLoRA) to save VRAM on T4")
args = parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID     = "google/medgemma-4b-it"
ROOT         = Path(__file__).parent.parent   # project root regardless of cwd
CHEXPERT_DIR = ROOT / "chexpert_plus"
OPENI_DIR    = ROOT / "openi"
DATA_DIR     = args.images or (OPENI_DIR if args.openi else CHEXPERT_DIR)
SUBSET_CSV   = ROOT / "smoke_subset.csv"
OUTPUT_DIR   = ROOT / "smoke_output"
DEVICE       = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_EPOCHS   = 2
BATCH_SIZE   = args.batch_size
LR           = 2e-4
MAX_SEQ_LEN  = args.max_seq_len
if DEVICE == "cuda":
    MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif DEVICE == "mps":
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32
USE_AUTOCAST = (DEVICE == "cuda") or (DEVICE == "mps" and args.mps_autocast)

PROMPT_TEMPLATE = (
    "You are an expert radiologist. Analyze this chest X-ray and write a "
    "structured radiology report with Findings and Impression sections."
)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}  Max seq len: {MAX_SEQ_LEN}  Single view: {args.single_view}")
print(f"Model dtype: {MODEL_DTYPE}")
if DEVICE == "mps" and not args.mps_autocast:
    print("MPS autocast disabled for stability")

# Fail fast before loading the model if required files are missing
if not args.synthetic:
    if not SUBSET_CSV.exists():
        hint = "python smoke_test/filter_openi.py" if args.openi else "python smoke_test/filter_chexpert.py"
        raise FileNotFoundError(
            f"smoke_subset.csv not found at {SUBSET_CSV}\n"
            f"Run first:  {hint}\n"
            "Or use:     python smoke_test/smoke_train.py --synthetic"
        )
    if not DATA_DIR.exists():
        src = "https://openi.nlm.nih.gov" if args.openi else "https://aimi.stanford.edu/datasets/chexpert-plus"
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nDownload from {src}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class CheXpertSmokeDataset(Dataset):
    def __init__(self, csv_path: Path, dicom_root: Path, include_lateral: bool = True):
        df = pd.read_csv(csv_path).reset_index(drop=True)
        # Build combined report from CheXpert Plus section columns
        if "section_findings" in df.columns and "section_impression" in df.columns:
            df["_report"] = (df["section_findings"].fillna("") + "\n" + df["section_impression"].fillna("")).str.strip()
        elif "findings" in df.columns and "impression" in df.columns:
            df["_report"] = (df["findings"].fillna("") + "\n" + df["impression"].fillna("")).str.strip()
        else:
            df["_report"] = df["report"].fillna("").str.strip()
        self.df         = df[df["_report"].str.len() > 20].reset_index(drop=True)
        self.dicom_root = dicom_root
        self.include_lateral = include_lateral
        if len(self.df) > 0:
            col = "path_to_dcm" if "path_to_dcm" in self.df.columns else "path"
            sample_path = self.dicom_root / str(self.df.iloc[0][col])
            if not sample_path.exists():
                hint = "\nHint: if smoke_subset.csv was built from OpenI, run with --openi and --images /path/to/NLMCXR_png"
                raise FileNotFoundError(f"Sample file not found: {sample_path}{hint}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        col     = "path_to_dcm" if "path_to_dcm" in self.df.columns else "path"
        images  = [self._load(self.dicom_root / row[col])]
        # include lateral view when present
        if self.include_lateral and "path_lateral" in self.df.columns and pd.notna(row.get("path_lateral")):
            lat = self.dicom_root / row["path_lateral"]
            if lat.exists():
                images.append(self._load(lat))
        return images, row["_report"]

    def _load(self, path: Path):
        if path.suffix.lower() == ".png":
            return PILImage.open(path).convert("RGB")
        return dicom_to_pil(load_dicom(path))


class SyntheticDataset(Dataset):
    """Fallback dataset using existing dataset/ patient DICOMs."""
    # Repeat samples to have enough steps for a meaningful smoke test
    REPEAT = 4

    def __init__(self, dataset_root: Path):
        pairs = []
        for report_file in sorted(dataset_root.rglob("report.txt")):
            dcm = report_file.parent / "image.dcm"
            if dcm.exists():
                pairs.append((dcm, report_file.read_text().strip()))
        if not pairs:
            raise FileNotFoundError(f"No patient data found under {dataset_root}")
        self.pairs = pairs * self.REPEAT

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        dcm_path, report = self.pairs[idx]
        return [dicom_to_pil(load_dicom(dcm_path))], report


# ── Load model ─────────────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(MODEL_ID)
if args.load_in_4bit and DEVICE == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=MODEL_DTYPE,
        device_map={"": DEVICE},
    )
print(f"Model loaded  params={sum(p.numel() for p in model.parameters()):,}")

lora_cfg = LoraConfig(
    r=8,                                        # lower rank for smoke test speed
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],        # minimal targets; full run uses 7
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# ── Collate ────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    image_lists, reports = zip(*batch)
    conversations = [
        [
            {"role": "user", "content":
                [{"type": "image"} for _ in imgs] +
                [{"type": "text", "text": PROMPT_TEMPLATE}]
            },
            {"role": "assistant", "content": [{"type": "text", "text": report}]},
        ]
        for imgs, report in zip(image_lists, reports)
    ]
    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]
    inputs = processor(
        text=list(texts),
        images=list(image_lists),   # List[List[PIL]] — one inner list per sample
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {k: v.to(DEVICE) for k, v in inputs.items()}


# ── Data loader ────────────────────────────────────────────────────────────────
if args.synthetic:
    dataset = SyntheticDataset(ROOT / "dataset")
    print(f"[synthetic mode] Dataset size: {len(dataset)} samples (from dataset/)")
else:
    dataset = CheXpertSmokeDataset(SUBSET_CSV, DATA_DIR, include_lateral=not args.single_view)
    mode = "OpenI" if args.openi else "CheXpert Plus"
    print(f"[{mode}] Dataset size: {len(dataset):,}")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Single forward pass to verify no crash before the full loop
model.train()
batch = next(iter(loader))
autocast_ctx = torch.autocast(DEVICE, dtype=MODEL_DTYPE) if USE_AUTOCAST else nullcontext()
with autocast_ctx:
    outputs = model(**batch)
print(f"✅  Forward pass OK  loss={outputs.loss.item():.4f}")
assert torch.isfinite(outputs.loss), "Loss is NaN/Inf — pipeline broken"


# ── Training loop ──────────────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR)
losses    = []

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for step, batch in enumerate(loader):
        optimizer.zero_grad()
        autocast_ctx = torch.autocast(DEVICE, dtype=MODEL_DTYPE) if USE_AUTOCAST else nullcontext()
        with autocast_ctx:
            loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}  step {step+1:>4}  loss={loss.item():.4f}")

    avg = epoch_loss / len(loader)
    losses.append(avg)
    print(f"── Epoch {epoch+1} avg loss: {avg:.4f}\n")

assert losses[-1] < losses[0], "⚠  Loss did not decrease — check data/config"
print(f"✅  Loss decreased: {losses[0]:.4f} → {losses[-1]:.4f}")


# ── Save adapter ───────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
processor.save_pretrained(str(OUTPUT_DIR))
print(f"✅  Adapter saved to {OUTPUT_DIR}/")


# ── Reload & verify ────────────────────────────────────────────────────────────
base     = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=MODEL_DTYPE, device_map={"": DEVICE}
)
reloaded = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
reloaded.eval()
print("✅  Adapter reloaded successfully")


# ── Inference check ────────────────────────────────────────────────────────────
test_imgs, test_report = dataset[0]

conversation = [{"role": "user", "content":
    [{"type": "image"} for _ in test_imgs] +
    [{"type": "text", "text": PROMPT_TEMPLATE}]
}]
prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=[test_imgs], return_tensors="pt").to(DEVICE)

with torch.no_grad():
    ids = reloaded.generate(**inputs, max_new_tokens=200, do_sample=False)

generated = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n── Generated ─────────────────────────────────")
print(generated)
print("\n── Ground Truth ──────────────────────────────")
print(test_report)

assert len(generated.strip()) > 20, "⚠  Model generated empty output"
print("\n✅  Inference check passed")
