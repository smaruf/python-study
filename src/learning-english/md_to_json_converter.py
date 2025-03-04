import re
import json

def parse_markdown(filename):
    """
    Parses the specified markdown file to extract vocabulary words and their details.
    """
    vocab_entries = {}
    
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Regex pattern to capture each vocabulary block (assuming each vocab entry is separated by two newlines)
    vocab_blocks = content.strip().split("\n\n")
    
    for block in vocab_blocks:
        # Extract each section based on markdown structures
        title_match = re.search(r"###\s(.+)", block)
        
        if not title_match:
            continue
        
        word = title_match.group(1).strip()
        vocab_entries[word] = {}
        
        # Patterns to extract each detail assuming line starts with `- **FieldName:**`
        entries = re.findall(r"- \*\*([^*]+)\*\*:\s(.+)", block)
        for entry in entries:
            field_name, field_content = entry
            vocab_entries[word][field_name.strip()] = field_content.strip()

    return vocab_entries

def save_as_json(data, out_filename):
    """
    Saves the parsed data to a JSON file.
    """
    with open(out_filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    input_md_file = 'Vocabulary_Dictionary_with_Pronunciations.md'  # Set the path to your Markdown file
    output_json_file = 'vocab_data.json'  # Output JSON file name
    
    vocab_data = parse_markdown(input_md_file)
    save_as_json(vocab_data, output_json_file)
    print("Extraction and conversion to JSON completed. Data saved to", output_json_file)
