import csv
from pathlib import Path

data_dir = Path(__file__).parent

out1 = data_dir / "1860s"
out2 = data_dir / "1950s"
out1.mkdir(exist_ok=True)
out2.mkdir(exist_ok=True)

with (
    open(out1 / "corpus1.txt", "w", encoding="utf-8") as f1,
    open(out2 / "corpus2.txt", "w", encoding="utf-8") as f2,
):

    for uses_file in sorted(data_dir.glob("*/uses.csv")):
        with open(uses_file, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile, delimiter="\t")
            for row in reader:
                try:
                    date = int(row["date"])
                except (ValueError, KeyError):
                    continue

                context = row["context"].strip()
                line = f"{context}\n"

                if date < 1860:
                    f1.write(line)
                elif date > 1950:
                    f2.write(line)
