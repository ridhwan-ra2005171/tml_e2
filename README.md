# Watermark Forgery Attack

This is the code behind our leaderboard submission for the Watermark Forging task (Team XXV).

## What you need

Install the dependencies:

```bash
pip install torch torchvision numpy pillow scipy lpips opencv-python trustmark "numpy<2.0" "scipy>=1.11" "lightning>=2.0"
```

TrustMark grabs its own pretrained weights the first time you run it, so you don't need to download anything separately or commit them to the repo. Everything here runs fine on CPU — no GPU needed.

## Files you'll need

Drop `Dataset.zip` next to `task_template.py`. Once unzipped it should look like this:

| Path | What's in it |
|---|---|
| `Dataset/watermarked_sources/WM_1` … `WM_8` | 25 watermarked examples per scheme (8 schemes total, and we don't know upfront what any of them are) |
| `Dataset/clean_targets/1.png` … `200.png` | 200 clean images we need to forge watermarks onto |

The clean targets come in three sizes (128×128, 256×256, 512×512), and each `WM_x` folder's sources are all one native resolution matching its 25-image block in the targets (`WM_1` → images 1–25, `WM_2` → 26–50, and so on up to `WM_8` → 176–200).

## Running it

```bash
python task_template.py
```

Here's what happens when you run it:

1. Unzips the dataset (only needs to do this once)
2. Tries to identify each of the 8 watermark sets using TrustMark, checking all 3 of its public model variants
3. Any set it actually manages to identify gets re-embedded using TrustMark's real encoder
4. Everything else falls back to a calibrated copy attack
5. Forges all 200 target images
6. Spits out a quality report (LPIPS / S_qlt) and zips it all into `submission.zip`

## How it actually works

### Step 1 — figuring out what scheme each set uses

For every one of the 8 sets, we decode all 25 sources with each TrustMark variant. We only trust an identification if it clears all three of these bars:

| Check | Bar to clear | Why it matters |
|---|---|---|
| Present rate | ≥ 0.75 | how often TrustMark's own detector says "yep, there's a watermark here" |
| Consensus agreement | ≥ 0.82 | how consistently the sources decode to the same message (random noise usually lands around 0.5–0.75, real matches jump to 0.95+) |
| Round-trip check | ≥ 0.95 | encode the message we found into a clean image, decode it back, see if we get the same thing — cheap way to catch false positives |

If a set clears all three, we re-embed it with TrustMark's actual encoder and the message we recovered. This gets near-perfect detection with barely any visible quality loss.

### Step 2 — copy attack for everything else

For sets TrustMark can't identify, we fall back to a blackbox averaging attack (based on Yang et al.):

1. Take the average of the 25 watermarked sources at their native resolution, subtract the average clean image at that same resolution — that gives us a rough residual pattern
2. Blur that residual heavily and subtract the blur, which strips out any general color tint that isn't actually part of the watermark
3. Build a mask that favors regions where the pattern stays consistent across sources and downplays regions where it's just following the image content
4. Binary-search the injection strength per set so the average quality loss (LPIPS) lands right at our budget — as strong a watermark as we can get away with

This works well against content-agnostic watermarks, but it hits a wall with content-adaptive neural ones — those embed differently depending on the image, so simple averaging can't fully recover them.

### Other decoders we tried and ruled out

Besides TrustMark, we also tested `dwtDct`, `dwtDctSvd`, and `rivaGan` (all via `imwatermark`), both public HiDDeN checkpoints, and VideoSeal (v0.0 and v1.0) — none of them showed anything beyond random noise on any of the 8 sets. We also looked at the residuals in frequency space (FFT) to check for any obvious structured patterns visually, just in case something jumped out that the decoders missed.

## How long it takes

Roughly 10–20 minutes on GPU for the whole thing — most of that time goes into LPIPS checks while calibrating the copy attack's strength.

## What you get out of it

A `submission.zip` with all 200 forged images, ready to upload. While it runs, you'll see a table showing what happened with each set (variant tried, present rate, agreement, round-trip score, and the final verdict), plus a quality summary at the end.

## Best score so far

**0.439026** on the leaderboard — we managed to genuinely crack one scheme (`WM_7`, turned out to be TrustMark-Q) and re-embed it properly, with the copy attack covering the other seven we couldn't identify.
