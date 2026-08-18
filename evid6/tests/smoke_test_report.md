# EVID-6 smoke test report

Generated 2026-08-18 20:38, Python 3.12.8,
numpy 2.4.6.

The offline test exercises the entire CPU path — schema, fold leakage guard,
metadata passthrough, probes, learning curve, consistency scoring, abstention,
ladder rungs 1-2, exact pairing, transfer, relabel harness, budget logging,
every figure, the Tier B loader, and the image generators — on synthetic data.
No GPU and no COCO download required.

**All 24 sections passed.**

Two of them are negative controls rather than happy paths: section 12 feeds a
world where P2 is false and asserts the verdict reports "challenged", and
section 23 strips intervention metadata and asserts the threats table reports
FAILED. A test suite that only ever confirms success cannot catch a scoring
function that always says yes.

Run it with `python tests/smoke_test.py`.
