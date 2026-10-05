from pypdf import PdfReader, PdfWriter

SOURCE = "data/datafull_report.pdf"
OUTPUT = "data/tata_motors_pv_trimmed.pdf"

# 1-based, inclusive page ranges of the original report:
#  20-37   operating environment, PV and EV business, key highlights
#  154-165 Board's Report (financial highlights, FY26 sales)
#  263-283 Management Discussion & Analysis
RANGES = [(20, 37), (154, 165), (263, 283)]

reader = PdfReader(SOURCE)
writer = PdfWriter()
count = 0
for start, end in RANGES:
    for i in range(start - 1, end):
        writer.add_page(reader.pages[i])
        count += 1

with open(OUTPUT, "wb") as f:
    writer.write(f)

print(f"Saved {count} pages to {OUTPUT}")