# Evaluate saved Valen observations

`pyrualean valen-eval` evaluates the request/response pairs in an episode's
`supervision/` audit. It never starts a robot or executes policy code. Cached
evaluation needs only the package and the saved JSON; crop replay also needs Pillow.

## Compare thresholds and prepare labels

```bash
pyrualean valen-eval --audit runs/my-episode/supervision \
    --out runs/valen-baseline --thresholds 0.6 0.7 0.8 0.9
```

The output directory must be new. Source audits and earlier comparisons are never
overwritten. Successful visual checks contribute scores; controller errors and
missing observations are not extra votes. Outputs are:

- `summary.json`: counts, unknown rates and optional labelled metrics per threshold.
- `predictions.json`: per-sample questions, raw probabilities and abstention reasons.
- `labels.json`: an annotation template with sample/fact IDs, exact questions,
  image paths and initially null labels.

Annotate `label` as `yes`, `no`, or `unknown` from the permitted visual evidence.
Use unknown when the images cannot establish the fact. Leave unreviewed entries
null. These are human visual-fact labels, not model predictions or simulator truth.

```bash
pyrualean valen-eval --audit runs/my-episode/supervision \
    --labels runs/valen-baseline/labels.json --out runs/valen-labelled
```

Each label binds to the sample ID, fact ID and exact question; a changed question
cannot silently inherit old labels. Reports include a three-way confusion matrix,
false-yes/false-no counts, decisions on human-unobservable facts, and accuracy
among decided answers with known yes/no labels. Unlabelled accuracy/error metrics
are null. A lower unknown rate by itself does not establish greater accuracy or
safer actions. No threshold is selected or changed in live supervision.

## Replay questions and local crops

Start a compatible local Valen service separately. The explicit `--replay-url`
enables fresh predictions on saved images. Without it, only saved scores are read.

```bash
pyrualean valen-eval --audit runs/my-episode/supervision \
    --replay-url http://127.0.0.1:18001/predict \
    --variant variants.json --out runs/valen-crop-comparison
```

Example `variants.json` (replace the image basename with one in your audit):

```json
{
  "questions": {
    "left_holds_key": "Is the round key handle visibly between the left gripper fingers?"
  },
  "crops": {
    "RUN_ID-00012-1.png": [180, 140, 470, 400]
  }
}
```

Question overrides apply only where that fact appears. Crops use integer
`[left, top, right, bottom]` pixels and are selected per image basename, including
separate before/after images when needed. Full scenes and camera/observation
labels are retained; a labelled local crop is appended, not treated as another
observation. Out-of-bounds crops and unknown image/fact names are rejected.
Replay expects the images alongside the audit JSON, sharing a filesystem with
the Valen service. Requests, raw responses, crops and latency are saved in the
new directory. Original observations are immutable.

Changing “held” to “between the fingers” changes the predicate. Export and review
new labels for that wording instead of interpreting the comparison as automatic
evidence of improved grasp verification. Compare false passes and abstentions on
labelled observations before changing online questions, views or thresholds.
Visual evaluation alone does not measure robot task success.
