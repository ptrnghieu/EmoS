# Cross-condition diagnostic: summary and decision

Existing predictions only (Stage 1 outer runs, seed 0; every clip read by the fold model that never saw its actor;
7,438 clips, 91 actors). Full tables: `report.md`; script: `analysis/cross_condition_diagnostic.py`.

**Statement.** s1_noI reproduces audio-only and audio-visual judgements about as well as another group of ~10 raters
would, but not video-only judgements. Its weak reproduction of how judgements change from video-only to audio-visual
is mostly inherited from that video-only reading: where the video-only reading is good, the change is reproduced at
the human re-run level, and audio-visual disambiguation is found as often as a human re-run finds it.

**Decision gate → branch 2: the errors already appear in the video-only condition.** The next step is visual
perception / video-only distribution learning, not a cross-modal integration method. No cross-condition change that
s1_noI misses consistently, while reading both unimodal conditions well, was found.

## Diagnostic table (s1_noI)

"Human re-run" = agreement of a simulated independent re-run of the human study (same number of raters per clip and
condition) with the observed votes. It is a reference, not an upper bound: a model of the underlying distribution can
exceed it, as s1_noI does on A and AV. "Emotion prior" = mean human change of the intended emotion, other folds.
Brackets: 95% bootstrap over actors.

| Quantity | s1_noI | Emotion prior | Human re-run | Reading |
| --- | --- | --- | --- | --- |
| **Each condition** — JSD to the votes (lower is better) | | | | |
| video only | 0.142 [0.138, 0.146] | — | 0.096 | clearly worse than humans |
| audio only | 0.087 [0.086, 0.089] | — | 0.100 | at or above human level |
| audio-visual | 0.088 [0.085, 0.091] | — | 0.095 | at or above human level |
| top label among the raters' top, video only | 0.651 | — | 0.784 | clearly worse |
| top label among the raters' top, audio only / AV | 0.774 / 0.802 | — | 0.730 / 0.807 | at human level |
| **Change V → AV, all clips** | | | | |
| redistribution (correlation over clip × label) | 0.326 [0.306, 0.345] | 0.255 | 0.550 | barely above the prior |
| entropy change (Spearman) | 0.202 [0.177, 0.228] | 0.143 | 0.391 | barely above the prior |
| **Change V → AV, clips whose two unimodal readings are good (n = 1,800)** | | | | |
| redistribution (correlation) | 0.522 [0.489, 0.551] | 0.339 | 0.529 | at human level |
| entropy change (Spearman) | 0.416 [0.378, 0.455] | 0.289 | 0.376 | at human level |
| **Change A → AV, all clips** | | | | |
| redistribution (correlation) | 0.684 [0.667, 0.698] | 0.653 | 0.693 | near human level; direction close to the prior |
| entropy change (Spearman) | 0.434 [0.406, 0.460] | 0.300 | 0.457 | near human level |
| **Integration** (5,613 clips whose unimodal votes differ) | | | | |
| audio weight in AV ≈ (1 − w) V + w A | 0.49 [0.47, 0.51] | — | 0.41 (observed: 0.35) | model leans more on audio |
| recall of human AV disambiguation (AV agrees more than both) | 0.48 [0.45, 0.51] | — | 0.48 | at human level |

m3 (AV labels only) is worse on every unimodal row, most on audio only (JSD 0.116), and finds only 19% of the human
disambiguations; per-condition supervision fixes the audio reading and the disambiguation rate but not the video reading.

## Failure cases

![s1_noI failure cases](failure_cases.png)

The six clips (one per emotion) where both unimodal readings of s1_noI are close to the votes but its V → AV change
points most against the human one. Two patterns:

- **Humans resolve, the model averages.** Video-only and audio-only raters are split, audio-visual raters converge
  (1083 Anger: AV 73% Anger; 1033 Fear: AV 100% Fear; 1091 Disgust: AV 80% Disgust). s1_noI's AV stays as split as its
  unimodal readings.
- **Humans inherit the other channel's doubt, the model stays confident.** One channel is clear, the other mixed, and
  AV raters are less sure than the clear channel (1040 Happy: AV 70/30 Happy/Neutral; 1038 Neutral: AV spread). s1_noI's
  AV is more confident than humans.

These are the six largest misses among 479 candidates, with about 10 raters per condition, so part of each pattern is
rater sampling noise; the aggregate rows show that neither pattern is systematic (disambiguation recall equals the
human re-run's).

## Caveats

- The "both readings good" subset is chosen by the model's agreement with the observed votes, which favours the model
  on that subset; the human re-run and the prior on the same subset are shown for that reason.
- One seed; Qwen2.5-Omni-7B only; CREMA-D only.
- The mixture weight w is computed from the model's own unimodal readings; with a weaker video reading, a higher audio
  weight is expected and is not by itself an integration failure.
