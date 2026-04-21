import fitz
doc = fitz.open("../Sample-report-DUMMY-9.pdf")
for i, page in enumerate(doc):
    text = page.get_text("text")
    print(f"--- PAGE {i+1} ---")
    print(text[:500])
    print()