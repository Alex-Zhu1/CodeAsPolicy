# Recovery cases and contract revisions

## Insert key: missed grasp plus uncertain lock orientation

Observed case: the left gripper closed empty, and Valen could not establish the
lock-face orientation. The episode ended at step 57 of 300, before handover,
insertion, or turning. Those are two distinct problems.

A useful revised plan is:

1. Reinspect the thin key and gripper pose. Record the grasp as missed, not held.
2. Determine whether approaching the key can avoid disturbing the lock. If so,
   the lock-face question is not a prerequisite for that grasp. If the approach
   passes close to the lock, choose another approach or resolve that geometry.
3. Defer lock/keyhole alignment to pre-insertion. Do not assume a lock must face
   up merely because the previous question used that wording; derive the actual
   alignment relation from the task and scene.
4. Choose a different grasp point, approach, or hand using the observed key
   geometry. Inspect the resulting grasp before transport or donor release.
5. Complete handover only after receiver possession is established. At
   pre-insertion, restore the keyhole/alignment checks and obtain useful evidence.
6. Insert and turn only after their respective prerequisites are established.

This is a hypothesis-driven recovery strategy, not proof that the task will be
solved or a source of reusable grasp coordinates.

## Example contracts: transport now, alignment later

Assume Astra has assessed the new grasp using images, object motion and gripper
state, and the chosen transport path does not contact the lock. This example
deliberately makes a fresh visual alignment answer a hard prerequisite for
insertion; it is not a default for every task. Do not execute these declarations
consecutively without the indicated state changes and evidence reviews.

```python
facts = {
    "held": "Is the key held between the right gripper fingers?",
    "aligned": "Is the key blade aligned with the keyhole opening?",
}
watch(
    "transport verified key grasp toward pre-insertion pose",
    facts,
    alerts=[{"name": "key lost", "when": {"held": "no"}}],
)
defer_fact("aligned", until="insert", expected="yes",
           reason="Free-space transport does not require blade/keyhole alignment")
# An aligned=no/unknown result is recorded without blocking transport.
# held=no requests review. Ordinary held=unknown is information for Astra;
# it does not veto the following transport action or prove that the grasp failed.
```

At the endpoint intended to establish alignment, change the contract before the
final aligning motion, then inspect its report in a separate decision. Alignment
is now a prerequisite for the *next* operation, insertion:

```python
watch(
    "establish alignment before insertion",
    facts,
    alerts=[
        {"name": "key lost", "when": {"held": "no"}},
        {"name": "blade not aligned", "when": {"aligned": "no"}},
    ],
    critical=["aligned"],
)
# Perform the intended alignment action, then end the cell for review.
# Astra assesses possession using all permitted evidence. This example separately
# requires a fresh aligned=yes Valen answer before insertion; clear is insufficient.
# Inspect evidence()['facts']['aligned'] for the confirming observation/step.
# Only after that verification, in the next cell:
watch("insert", facts, requires={"aligned": "yes"},
      alerts=[{"name": "key lost", "when": {"held": "no"}}])
# An unresolved or stale aligned fact stops insertion before its first action.
```

The host still uses its configured probability threshold and unknown window.
Deferral changes when an explicit hard condition blocks work, not the definition
of truth. A normal critical review can return to Astra and a subsequent cell;
that handoff cannot clear the separate registered alignment prerequisite.

## Other cases

| Blockage | Reasonable recovery hypothesis | Condition that remains blocking |
|---|---|---|
| Cup-grasp retry blocked by uncertain drawer closure | Grasp/place the cup first if the drawer does not affect the approach; inspect closure later | Verified grasp before carrying the cup |
| Upper structure uncertain while arranging separate lower supports | Build independent supports and defer the future upper relationship | Existing load-bearing relationships affected by the action |
| Connector mating face occluded | Move to a visible pre-insertion pose or use a permitted alternate view | Alignment/clearance before insertion |
| A carried object's grasp is uncertain | Inspect/secure/regrasp it before dependent transport | Do not defer grasp integrity just to keep moving |
| Missing or stale monitor image | Restore observation delivery or choose an available supported observation path | No fabricated visual evidence or silent monitor disable |
| Episode already finalized with stuck | Diagnose and prepare a revised plan for an authorized new episode | Do not issue further actions to the ended episode |

## Recovery decision record

```text
Blockage: actual missed grasp + uncertain later lock alignment.
Current state: key still on table; no handover occurred.
Reclassification: lock alignment deferred; key acquisition remains current.
Candidate: alternative grasp from an observed accessible side.
Expected progress: lift key with verified possession.
Active guards: constraints relevant to this approach, then held during transport.
Deferred gate: restore and establish blade/keyhole alignment before insertion.
Budget: existing episode remainder; one attempt then inspect changed evidence.
If no progress: revisit grasp geometry or another feasible candidate; do not
repeat unchanged code or convert unknown into success.
```

Keep these causal records concise. The host audits registered deferrals and checks
their evidence before motions in the named phase; it cannot infer omitted
dependencies or recognize an incorrectly named phase from Python or geometry.
