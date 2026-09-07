# ICRA 2027 drafts: two alternative framings

These are **alternative drafts of the same study**, not two papers to submit
simultaneously. Both are anonymous, English, two-column manuscripts using the
official PaperCept `ieeeconf` class. The detailed working manuscript is preserved
at `../CODE_AS_A_LEARNING_MACHINE_MANUSCRIPT.md`.

## Read

- [A: Learning to Open and Remove](learning_to_open_and_remove.pdf)
- [B: RoboEvolve](roboevolve.pdf)
- [日本語の比較・推奨](COMPARISON_JA.md)
- [Internal claim-to-evidence map](EVIDENCE_MAP.md)

## Build and validate

From this directory:

```sh
make all
make check
```

Dependencies: Python with NumPy, Matplotlib, seaborn and Pillow; TeX Live with
`latexmk`, `pdflatex`, BibTeX, `IEEEtran.bst`, `flushend`, and the standard
packages in `preamble.tex`; Poppler (`pdfinfo`, `pdffonts`, `pdftotext`).
The build reads tracked figures and numeric reports. It does not import robot
control modules, contact hardware, or need the private conversation archive.
Only the optional command `python extract_evidence.py` requires the archived
cap capture manifests and historical Git objects. It is read-only with respect
to the robot and exports only selected numbers and hashes to `evidence.json`.

`make_figures.py` regenerates plots from tracked measured values. The exported
whole-frame cap image is included, so raw captures are unnecessary for ordinary
builds. `figure_provenance.json` records input hashes; `evidence.json` records
cap pose measurements and historical code hashes. Build logs and automated
validation results live under ignored `build/`.

## Formatting and disclosure

The initial-submission call specifies at most eight pages including references,
double-column formatting, double-anonymous review, and generative-AI disclosure:
https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
(checked 2026-09-07). Some other conference pages contain different final-paper
instructions; these drafts follow the initial-submission call.

The unmodified class was downloaded from:
https://ras.papercept.net/conferences/support/files/ieeeconf.zip
See the copyright/license comments in `ieeeconf.cls`. No manual margin or font
shrinking is used. `flushend` balances the last page. Actual submission and
PaperCept compliance validation are not performed by this build.

The PDF authors are anonymous and metadata author is blank. The internal
evidence map and source repository are **not an anonymous supplement**. Do not
upload them unchanged. Authors must review factual claims, disclosure, author
list, model/version reporting, ethics/safety requirements and references before
submission. No current robot performance or prospective ablation is implied.
