import re
# Search for a pattern in a string
text = "The quick brown fox jumps over the lazy dog"
match = re.search(r"brown", text)
if match:
  print("Match found!")
# Find all occurrences of a pattern in a string
text = "The quick brown fox jumps over the lazy dog"
matches = re.findall(r"the", text, re.IGNORECASE)
print(matches)
# Replace all occurrences of a pattern in a string
text = "The quick brown fox jumps over the lazy dog"
new_text = re.sub(r"the", "that", text, flags=re.IGNORECASE)
print(new_text)
