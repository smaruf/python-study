import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
from gtts import gTTS
import os
import re

class TranslationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Translation Dictionary")
        
        self.leading_language = tk.StringVar()
        self.word_list = []
        
        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        # Dropdown for selecting leading language
        ttk.Label(self.root, text="Select Leading Language:").grid(column=0, row=0, padx=10, pady=10)
        self.language_selector = ttk.Combobox(self.root, textvariable=self.leading_language)
        self.language_selector.grid(column=1, row=0, padx=10, pady=10)
        
        # Listbox to display words
        self.word_listbox = tk.Listbox(self.root, width=50)
        self.word_listbox.grid(column=0, row=1, columnspan=2, padx=10, pady=10)
        self.word_listbox.bind('<<ListboxSelect>>', self.on_word_select)
        
        # Button to swap order of translations
        self.swap_button = ttk.Button(self.root, text="Swap Translations", command=self.swap_translations)
        self.swap_button.grid(column=0, row=2, padx=10, pady=10)
        
        # Button to play sound
        self.sound_button = ttk.Button(self.root, text="Play Sound", command=self.play_sound)
        self.sound_button.grid(column=1, row=2, padx=10, pady=10)
    
    def load_data(self):
        # Load and parse the markdown file
        filename = filedialog.askopenfilename(title="Select the Dictionary File", filetypes=[("Markdown files", "*.md")])
        if filename:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.readlines()
            
            # Extract words and translations
            pattern = re.compile(r'\|(.+?)\|(.+?)\|(.+?)\|(.+?)\|')
            for line in content:
                match = pattern.match(line)
                if match:
                    self.word_list.append(match.groups())
            
            # Set available languages
            if self.word_list:
                self.language_selector['values'] = ['Polish', 'Arabic', 'Bangla', 'English']
                self.language_selector.current(0)
                self.update_word_listbox()
    
    def update_word_listbox(self):
        self.word_listbox.delete(0, tk.END)
        leading_index = self.language_selector.current()
        
        for words in self.word_list:
            self.word_listbox.insert(tk.END, words[leading_index])
    
    def on_word_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            selected_word = self.word_list[index]
            leading_index = self.language_selector.current()
            
            self.selected_translations = selected_word
            self.play_text = selected_word[leading_index]
    
    def swap_translations(self):
        if hasattr(self, 'selected_translations'):
            leading_index = self.language_selector.current()
            self.selected_translations = self.selected_translations[leading_index:] + self.selected_translations[:leading_index]
            self.update_word_listbox()
    
    def play_sound(self):
        if hasattr(self, 'play_text'):
            tts = gTTS(text=self.play_text, lang='en')
            tts.save('temp.mp3')
            os.system('mpg321 temp.mp3')
            os.remove('temp.mp3')

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslationApp(root)
    root.mainloop()
