"""Render README measurements only from this run's completed CI artifacts."""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

root, destination = Path("artifacts"), Path("reports")
destination.mkdir(exist_ok=True)


def test_counts(path):
    tree = ET.parse(path).getroot()
    suites = [tree] if tree.tag == "testsuite" else list(tree.iter("testsuite"))
    totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))
    assert totals["tests"] and not any(totals[key] for key in ("failures", "errors", "skipped"))
    return totals


tests = {"c": test_counts(root / "c-junit.xml"), "python": test_counts(root / "python-junit.xml")}
python_coverage = json.loads((root / "python-coverage.json").read_text())["totals"]
c_coverage = json.loads((root / "c-coverage.json").read_text())
bench = json.loads((root / "measurements/bench.json").read_text())
model = json.loads((root / "model/comparison.json").read_text())
assert bench["passed"] and model["training"]["final_loss"] < model["training"]["initial_loss"]
url = f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
report = dict(workflow_url=url, source_commit=os.environ["GITHUB_SHA"], tests=tests,
              python_coverage=python_coverage, c_coverage=c_coverage, fault_bench=bench, model=model)
(destination / "final.json").write_text(json.dumps(report, indent=2) + "\n")
shutil.copyfile(root / "environment.txt", destination / "environment.txt")
for name in ("clean", "dropped_frames", "bad_crc", "stuck_sensor", "spikes"):
    shutil.copytree(root / "measurements" / name / "analysis", destination / name, dirs_exist_ok=True)
float_model, int8_model = model["models"]["float"], model["models"]["int8"]
size_ratio = int8_model["bytes"] / float_model["bytes"]
latency_ratio = int8_model["latency"]["median_us"] / float_model["latency"]["median_us"]
lines = [
    f"Measured by [GitHub Actions]({url}) on source commit `{os.environ['GITHUB_SHA']}`.", "",
    f"**{tests['c']['tests'] + tests['python']['tests']} tests passed**: "
    f"{tests['c']['tests']} CTest cases and {tests['python']['tests']} pytest cases; none skipped.", "",
    f"Python statement and branch coverage: **{python_coverage['percent_covered']:.2f}%**. "
    f"C line coverage: **{c_coverage['line_percent']:.2f}%**; "
    f"C branch coverage: **{c_coverage['branch_percent']:.2f}%**.", "",
    "The fault bench detected dropped frames, bad CRC, stuck sensor, and spikes, "
    "with no alerts on its clean control.", "",
    "| ONNX model | File bytes | Median latency | p95 latency | Precision | Recall | F1 | ROC AUC |",
    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name, item in model["models"].items():
    quality, latency = item["quality"], item["latency"]
    lines.append(f"| {name} | {item['bytes']} | {latency['median_us']:.3f} us | "
                 f"{latency['p95_us']:.3f} us | {quality['precision']:.4f} | "
                 f"{quality['recall']:.4f} | {quality['f1']:.4f} | {quality['roc_auc']:.4f} |")
lines += [
    "",
    f"Int8/float size ratio: **{size_ratio:.3f}**. Int8/float median latency ratio: "
    f"**{latency_ratio:.3f}** (above 1 means slower). Training took "
    f"**{model['training']['seconds']:.2f} s** on this hosted CPU.", "",
    f"Evaluation: **{len(model['outcomes'])} feature windows** from separate seeded runs. "
    f"Both models use the same float-calibrated threshold `{model['threshold']:.6g}`. "
    f"Quantization caused **{model['int8_losses']} newly wrong decisions** and "
    f"**{model['int8_gains']} newly correct decisions**; "
    f"**{model['decision_disagreements']} total disagreements**.", "",
    "| Injected class | Float alarm rate | Int8 alarm rate |",
    "| :--- | ---: | ---: |",
]
for kind in float_model["by_class"]:
    lines.append(f"| {kind} | {float_model['by_class'][kind]['alarm_rate']:.2%} | "
                 f"{int8_model['by_class'][kind]['alarm_rate']:.2%} |")
lines += ["", "For normal windows, alarm rate is the false positive rate. For fault windows, "
          "it is recall. Low amplitude faults may be missed by either model.", ""]
if model["int8_loss_examples"]:
    lines += ["Cases where int8 loses a correct float decision:", ""]
    for item in model["int8_loss_examples"]:
        lines.append(f"- `{item['kind']}`, seed {item['seed']}, window {item['window']}: "
                     f"float score {item['float_score']:.6g}, int8 score {item['int8_score']:.6g}.")
else:
    lines += ["No newly wrong int8 decisions occurred in this finite evaluation set."]
lines += [
    "", "Timing measures batch-one ONNX inference only, with one thread, 50 warmups, "
    "and seven alternating rounds of 500 calls per model. It excludes feature extraction "
    "and acquisition. Hosted x86 timing is not a microcontroller measurement.", "",
    "Raw outcomes, confusion matrices, quantization details, seeds, and coverage are in "
    "[reports/final.json](reports/final.json). Per-run examples: "
    "[clean](reports/clean/measurement.md), [dropped frames](reports/dropped_frames/measurement.md), "
    "[bad CRC](reports/bad_crc/measurement.md), [stuck sensor](reports/stuck_sensor/measurement.md), "
    "and [spikes](reports/spikes/measurement.md). ONNX files, checkpoints, and coverage HTML "
    "are available in the workflow artifacts.", "",
    "This section and the reports are committed after all verification jobs pass. "
    "That generated report commit does not trigger a new run; the source commit above "
    "identifies the code measured.",
]
readme = Path("README.md")
before, tail = readme.read_text().split("<!-- RESULTS:START -->")
_, after = tail.split("<!-- RESULTS:END -->")
readme.write_text(before + "<!-- RESULTS:START -->\n" + "\n".join(lines)
                  + "\n<!-- RESULTS:END -->" + after)
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
    output.write("\n".join(lines) + "\n")
