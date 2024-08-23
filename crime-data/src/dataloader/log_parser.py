import argparse
import pathlib
import pickle
import re


def extract_metrics(texto, subset):
    pattern_epoch = rf"INFO:root:################ {subset} epoch (\d+) ###################\n(.*?)(?=INFO:root:################ {subset} epoch|\Z)"
    pattern_metrics = {
        "mean_losses": rf"{subset} mean losses tensor\(([\d\.]+), device='cuda:\d+'\)",
        "mean_errors": rf"{subset} mean errors ([\d\.]+)",
        "mean_MRR": rf"{subset} mean MRR ([\d\.]+)",
        "mean_MAP": rf"{subset} .+ mean MAP ([\d\.]+)",
        "microavg_precision": rf"{subset} measures microavg - precision ([\d\.]+)",
        "microavg_recall": rf"{subset} measures microavg .+ - recall ([\d\.]+)",
        "microavg_f1": rf"{subset} measures microavg .+ - f1 ([\d\.]+)",
        "class0_precision": rf"{subset} measures for class 0 - precision ([\d\.]+)",
        "class0_recall": rf"{subset} measures for class 0 .+ - recall ([\d\.]+)",
        "class0_f1": rf"{subset} measures for class 0 .+ - f1 ([\d\.]+)",
        "class1_precision": rf"{subset} measures for class 1 - precision ([\d\.]+)",
        "class1_recall": rf"{subset} measures for class 1 .+ - recall ([\d\.]+)",
        "class1_f1": rf"{subset} measures for class 1 .+ - f1 ([\d\.]+)",
    }

    epochs = re.findall(pattern_epoch, texto, re.DOTALL)

    results = {}
    for epoch, content in epochs:
        epoch = int(epoch)
        results[epoch] = {}
        for key, pattern in pattern_metrics.items():
            match = re.search(pattern, content)
            if match:
                results[epoch][key] = float(match.group(1))

    return results


parser = argparse.ArgumentParser()


parser.add_argument(
    "--file_name",
    default=None,
    type=str,
    help="File name for log to parse.",
)

args = parser.parse_args()

log_path = pathlib.Path(f"data/log/{args.file_name}")

output_path = pathlib.Path("data/log/")
output_path.mkdir(exist_ok=True, parents=True)

with open(log_path, "r") as f:
    log_file = f.read()


# Get parameters

param_pattern = r"INFO:root:\*\*\* PARAMETERS \*\*\*\nINFO:root:({[\s\S]*?})\nINFO:root:"

param_match = re.search(param_pattern, log_file)
param_text = param_match.group(1)
param_text = re.sub(r"'sp_args': <utils\.Namespace object at 0x[a-fA-F0-9]+>,\n", "", param_text)
param_text = param_text.replace("\n", "")
param_text = param_text.replace(" ", "")

parameters = eval(param_text)

# Get metrics

metrics_train = extract_metrics(log_file, "TRAIN")
metrics_val = extract_metrics(log_file, "VALID")
metrics_test = extract_metrics(log_file, "TEST")

with open(output_path / f"{args.file_name}-train-parsed.pickle", "wb") as f:
    pickle.dump(metrics_train, f)

with open(output_path / f"{args.file_name}-val-parsed.pickle", "wb") as f:
    pickle.dump(metrics_val, f)

with open(output_path / f"{args.file_name}-test-parsed.pickle", "wb") as f:
    pickle.dump(metrics_test, f)

with open(output_path / f"{args.file_name}-params-parsed.pickle", "wb") as f:
    pickle.dump(parameters, f)
