import csv

input_path = "corpus/semeval2020_ulscd_eng/truth.txt"
output_path = "corpus/semeval2020_ulscd_eng/truth.csv"

with open(input_path, encoding="utf-8") as infile, \
     open(output_path, "w", newline="", encoding="utf-8") as outfile:

    writer = csv.DictWriter(outfile, fieldnames=["lemma", "change_graded"])
    writer.writeheader()

    for line in infile:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        lemma = parts[0].split("_")[0]
        writer.writerow({"lemma": lemma, "change_graded": parts[1]})

print(f"Written to {output_path}")
