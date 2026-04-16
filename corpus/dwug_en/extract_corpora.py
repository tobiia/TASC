import csv
from pathlib import Path

data_dir = Path("corpus/dwug_en/data")

with open("corpus1.txt", "w", encoding="utf-8") as f1, open(
    "corpus2.txt", "w", encoding="utf-8"
) as f2:

    for uses_file in sorted(data_dir.glob("*/uses.csv")):
        with open(uses_file, newline="", encoding="utf-8") as infile:
            reader = csv.DictReader(infile, delimiter="\t")
            for row in reader:
                try:
                    date = int(row["date"])
                except (ValueError, KeyError):
                    continue

                lemma = row["lemma"]
                context = row["context"].strip()
                line = f"{context}\n"

                if date < 1860:
                    f1.write(line)
                elif date > 1950:
                    f2.write(line)
