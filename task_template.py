import sys
import json
import math
import argparse
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset


# config
BASE        = Path(__file__).parent
TARGET_CKPT = BASE / "target_model" / "weights.safetensors"
SUSPECT_DIR = BASE / "suspect_models"
DATA_ROOT   = BASE / "cifar100"
TRAIN_IDX   = BASE / "target_model" / "train_main_idx.json"
OUTPUT_CSV  = BASE / "SUBMISSION.csv"

BASE_URL = "http://34.63.153.158"                   
API_KEY  = "APIKEY"       
TASK_ID  = "19-stolen-model-detection"              
PROBE_SIZE  = 2000   #CIFAR-100 test images used as probe set
TRAIN_PROBE = 2000   #target's exact training images (from train_main_idx.json)
BATCH_SIZE  = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

# signal weights (these sum to 1)
W_CKA  = 0.25   # layer-4 representation similarity
W_OA   = 0.25   # output-distribution agreement (1 - JSD)
W_LC   = 0.15   # logit correlation
W_PA   = 0.15   # top-1 prediction agreement
W_MEM  = 0.20   # training-set membership signal (train_main_idx.json)


# git-lfs resolver, from huggingface
HF_REPO     = "SprintML/tml26_task2"
LFS_URL     = f"https://huggingface.co/{HF_REPO}.git/info/lfs/objects/batch"
LFS_HEADERS = {"Content-Type": "application/vnd.git-lfs+json",
               "Accept":       "application/vnd.git-lfs+json"}

def parse_lfs_pointer(path):
    try:
        text = Path(path).read_text()
        if "git-lfs" not in text:
            return None
        oid  = next(l.split("sha256:")[1].strip()
                    for l in text.splitlines() if l.startswith("oid"))
        size = int(next(l.split()[1]
                    for l in text.splitlines() if l.startswith("size")))
        return oid, size
    except Exception:
        return None

def download_lfs_file(path):
    ptr = parse_lfs_pointer(path)
    if ptr is None:
        return
    oid, size = ptr
    print(f"  LFS download: {Path(path).name} ({size/1e6:.1f} MB)...")
    payload = {"operation": "download", "transfers": ["basic"],
               "objects": [{"oid": oid, "size": size}]}
    resp = requests.post(LFS_URL, json=payload, headers=LFS_HEADERS, timeout=30)
    resp.raise_for_status()
    href = resp.json()["objects"][0]["actions"]["download"]["href"]
    with requests.get(href, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

def resolve_lfs_files():
    files = [TARGET_CKPT] + sorted(
        SUSPECT_DIR.glob("*.safetensors"),
        key=lambda p: int(p.stem.replace("suspect_", "")))
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Checking {f.name}...")
        try:
            download_lfs_file(f)
        except Exception as e:
            print(f"  [WARN] Failed to resolve {f.name}: {e}")

print("Resolving Git-LFS pointers...")
resolve_lfs_files()
print("All files resolved.")


# model factory
def make_model():
    m = resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 100)
    return m

def load_model(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        state_dict = load_file(str(path), device="cpu")
    else:
        state_dict = torch.load(str(path), map_location="cpu")
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        if len(keys) == 1 and isinstance(state_dict[keys[0]], dict):
            state_dict = state_dict[keys[0]]
    m = make_model()
    try:
        m.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        m.load_state_dict(state_dict, strict=False)
    m.eval()
    return m.to(DEVICE)


# load datasets
print("\nStarted at:", datetime.now())
print("Loading datasets...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# test-set probe (held-out, never seen during target training)
full_test = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform)
g = torch.Generator(); g.manual_seed(42)
probe_idx    = torch.randperm(len(full_test), generator=g)[:PROBE_SIZE].tolist()
probe_ds     = Subset(full_test, probe_idx)
probe_loader = DataLoader(probe_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# membership probe: exact training samples the target was trained on
full_train = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
with open(TRAIN_IDX) as f:
    train_main_idx = json.load(f)
g2 = torch.Generator(); g2.manual_seed(42)
mperm         = torch.randperm(len(train_main_idx), generator=g2)[:TRAIN_PROBE].tolist()
member_idx    = [train_main_idx[i] for i in mperm]
member_ds     = Subset(full_train, member_idx)
member_loader = DataLoader(member_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# signal helpers
def collect_logits(m, loader=None):
    loader = loader if loader is not None else probe_loader
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(m(x.to(DEVICE)).cpu())
    return torch.cat(out).float()

def collect_correct_conf(m, loader):
    confs = []
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(m(x.to(DEVICE)), dim=1).cpu()
            confs.append(p[torch.arange(len(y)), y])
    return torch.cat(confs).float()

def collect_layer4_acts(m):
    acts, buf = [], []
    def _hook(_, __, out):
        buf.append(out.mean(dim=[2, 3]).detach().cpu())
    h = m.layer4.register_forward_hook(_hook)
    with torch.no_grad():
        for x, _ in probe_loader:
            m(x.to(DEVICE)); acts.append(buf[-1])
    h.remove()
    return torch.cat(acts).float()

def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    n = X.shape[0]
    hsic_xy = ((X.T @ Y) ** 2).sum() / n**2
    hsic_xx = ((X.T @ X) ** 2).sum() / n**2
    hsic_yy = ((Y.T @ Y) ** 2).sum() / n**2
    return (hsic_xy / (hsic_xx * hsic_yy).sqrt().clamp(1e-8)).item()

def output_agreement(probs_a, probs_b):
    M     = 0.5 * (probs_a + probs_b)
    kl_am = (probs_a * (probs_a.clamp(1e-9).log() - M.clamp(1e-9).log())).sum(-1)
    kl_bm = (probs_b * (probs_b.clamp(1e-9).log() - M.clamp(1e-9).log())).sum(-1)
    return 1.0 - (0.5 * (kl_am + kl_bm)).mean().item() / math.log(2)

def logit_correlation(logits_a, logits_b):
    a = logits_a - logits_a.mean(1, keepdim=True)
    b = logits_b - logits_b.mean(1, keepdim=True)
    corr = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(1e-8)).mean().item()
    return (corr + 1.0) / 2.0

def prediction_agreement(preds_a, probs_b):
    return (preds_a == probs_b.argmax(1)).float().mean().item()

def membership_signal(target_member_conf, target_test_mean, suspect):
    s_member_conf = collect_correct_conf(suspect, member_loader)
    s_test_conf   = collect_correct_conf(suspect, probe_loader)
    target_gap  = target_member_conf.mean().item() - target_test_mean
    suspect_gap = s_member_conf.mean().item() - s_test_conf.mean().item()
    if abs(target_gap) < 1e-6:
        return 0.5
    return max(0.0, 1.0 - abs(target_gap - suspect_gap) / abs(target_gap))


# load target model and pre-compute its signals
print(f"Using device: {DEVICE}")
print("Loading target model...")
target             = load_model(TARGET_CKPT)
target_logits      = collect_logits(target)
target_probs       = F.softmax(target_logits, dim=1)
target_preds       = target_probs.argmax(1)
target_layer4      = collect_layer4_acts(target)
target_member_conf = collect_correct_conf(target, member_loader)
target_test_mean   = collect_correct_conf(target, probe_loader).mean().item()
print(f"  Train-set confidence:  {target_member_conf.mean():.4f}")
print(f"  Test-set confidence:   {target_test_mean:.4f}")
print(f"  Memorisation gap:      {target_member_conf.mean().item() - target_test_mean:.4f}")


# score all suspect models
suspect_files = sorted(
    SUSPECT_DIR.glob("*.safetensors"),
    key=lambda p: int(p.stem.replace("suspect_", "")),
)
assert len(suspect_files) > 0, f"No .safetensors files found in {SUSPECT_DIR}"
print(f"\nFound {len(suspect_files)} suspect models. Scoring...\n")

ids, raw_scores, all_signals = [], [], []

for sf in suspect_files:
    model_id = int(sf.stem.replace("suspect_", ""))
    cka = oa = lc = pa = mem = 0.0
    try:
        suspect     = load_model(sf)
        susp_logits = collect_logits(suspect)
        susp_probs  = F.softmax(susp_logits, dim=1)
        susp_layer4 = collect_layer4_acts(suspect)

        cka = linear_cka(target_layer4, susp_layer4)
        oa  = output_agreement(target_probs, susp_probs)
        lc  = logit_correlation(target_logits, susp_logits)
        pa  = prediction_agreement(target_preds, susp_probs)
        mem = membership_signal(target_member_conf, target_test_mean, suspect)

        score = W_CKA*cka + W_OA*oa + W_LC*lc + W_PA*pa + W_MEM*mem
        del suspect
    except Exception as e:
        print(f"  [WARN] model {model_id} failed: {e}")
        score = 0.0

    ids.append(model_id)
    raw_scores.append(score)
    all_signals.append([cka, oa, lc, pa, mem])
    print(f"  [{model_id:>3}] cka={cka:.3f} oa={oa:.3f} lc={lc:.3f} "
          f"pa={pa:.3f} mem={mem:.3f} -> {score:.4f}")


# rank-normalise to [0, 1]
def rank_pct(values):
    arr   = np.asarray(values, dtype=float)
    order = arr.argsort()
    out   = np.zeros_like(arr)
    out[order] = np.arange(len(arr)) / max(len(arr) - 1, 1)
    return out

final_scores = rank_pct(raw_scores)

df = pd.DataFrame({"id": ids, "score": final_scores})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

diag = pd.DataFrame(all_signals, columns=["cka", "oa", "lc", "pa", "mem"])
diag.insert(0, "id", ids)
diag["raw_score"]   = raw_scores
diag["final_score"] = final_scores
diag.to_csv(str(OUTPUT_CSV).replace(".csv", "_diagnostics.csv"), index=False)
print("Diagnostics saved.")


# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser()
args = parser.parse_args()

if not OUTPUT_CSV.exists():
    die(f"File not found: {OUTPUT_CSV}")

SUBMIT = True   #set to False to skip submission

if SUBMIT:
    print("Submitting at:", datetime.now())
    try:
        with open(OUTPUT_CSV, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files={"file": (OUTPUT_CSV.name, f, "application/csv")},
                timeout=(60, 600),
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413).")
        resp.raise_for_status()
        print("Successfully submitted.")
        print("Server response:", body)
        if body.get("submission_id"):
            print(f"Submission ID: {body['submission_id']}")
    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)