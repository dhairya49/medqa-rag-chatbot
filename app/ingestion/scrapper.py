import requests
from bs4 import BeautifulSoup
import time
import json
from tqdm import tqdm

BASE_URL = "https://www.drugs.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------------
# Step 1: Get drug links (A-Z index)
# -----------------------------------
def get_drug_links():
    links = []
    
    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.drugs.com/alpha/{letter}.html"
        res = requests.get(url, headers=HEADERS)

        soup = BeautifulSoup(res.text, "html.parser")

        for a in soup.select("ul.ddc-list-column-2 li a"):
            href = a.get("href")
            if href and href.startswith("/"):
                full_url = "https://www.drugs.com" + href
                links.append(full_url)

        time.sleep(0.5)

    return list(set(links))


# -----------------------------------
# Step 2: Extract drug page content
# -----------------------------------
def scrape_drug_page(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    title = soup.find("h1")
    drug_name = title.text.strip() if title else "Unknown"

    content = {
        "drug_name": drug_name,
        "uses": "",
        "dosage": "",
        "warnings": "",
        "side_effects": ""
    }

    # Sections are usually under h2 headings
    for section in soup.find_all(["h2", "h3"]):
        heading = section.text.lower()

        text_block = []
        for sib in section.find_next_siblings():
            if sib.name in ["h2", "h3"]:
                break
            text_block.append(sib.get_text(" ", strip=True))

        full_text = " ".join(text_block)

        if "uses" in heading:
            content["uses"] = full_text
        elif "dosage" in heading:
            content["dosage"] = full_text
        elif "warning" in heading:
            content["warnings"] = full_text
        elif "side effects" in heading:
            content["side_effects"] = full_text

    return content


# -----------------------------------
# Step 3: Structure-aware chunking
# -----------------------------------
def chunk_drug_data(drug):
    chunks = []

    def add_chunk(chunk_type, text):
        if text and len(text.strip()) > 20:
            chunks.append({
                "drug_name": drug["drug_name"],
                "type": chunk_type,
                "content": text.strip()
            })

    add_chunk("uses", drug["uses"])
    add_chunk("dosage", drug["dosage"])
    add_chunk("warnings", drug["warnings"])
    add_chunk("side_effects", drug["side_effects"])

    return chunks


# -----------------------------------
# Step 4: Split long chunks
# -----------------------------------
def split_chunks(chunks, max_chars=600):
    final_chunks = []

    for chunk in chunks:
        text = chunk["content"]

        if len(text) <= max_chars:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(text), max_chars):
                new_chunk = chunk.copy()
                new_chunk["content"] = text[i:i+max_chars]
                final_chunks.append(new_chunk)

    return final_chunks


# -----------------------------------
# Step 5: Main pipeline
# -----------------------------------
def build_dataset(limit=50):
    links = get_drug_links()
    links = links[:limit]

    all_chunks = []

    for url in tqdm(links):
        try:
            drug = scrape_drug_page(url)
            chunks = chunk_drug_data(drug)
            chunks = split_chunks(chunks)

            all_chunks.extend(chunks)

            time.sleep(1)  # be respectful

        except Exception as e:
            print(f"Error: {url}", e)

    return all_chunks


# -----------------------------------
# Save output
# -----------------------------------
if __name__ == "__main__":
    data = build_dataset(limit=30)

    with open("drugscom_chunks.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} chunks.")