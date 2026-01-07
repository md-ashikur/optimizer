#!/usr/bin/env python3
import csv
import sys
import os

def clean_csv(infile, outfile=None, max_empty=3):
    if outfile is None:
        base, ext = os.path.splitext(infile)
        outfile = base + '.cleaned' + ext
    removed = 0
    total = 0
    with open(infile, newline='', encoding='utf-8-sig') as inf, open(outfile, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf)
        try:
            header = next(reader)
        except StopIteration:
            print('Input file is empty')
            return 0, 0, outfile
        writer.writerow(header)
        for row in reader:
            total += 1
            empty_count = sum(1 for cell in row if (cell is None) or (str(cell).strip() == ''))
            if empty_count > max_empty:
                removed += 1
            else:
                writer.writerow(row)
    return total, removed, outfile

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python Code/clean_csv.py "path/to/file.csv" [max_empty]')
        sys.exit(1)
    infile = sys.argv[1]
    max_empty = int(sys.argv[2]) if len(sys.argv) >= 3 else 3
    total, removed, outfile = clean_csv(infile, max_empty=max_empty)
    kept = total - removed
    print(f'Processed {total} rows. Removed {removed}. Kept {kept}. Output: {outfile}')
