#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk

# Create the main window
window = tk.Tk()
window.title('My Window')
window.geometry('500x300')

# String variable for the Label text
selected_item_var = tk.StringVar()

# Create a Label to display the selected item
label = tk.Label(
    window,
    bg='green',
    fg='yellow',
    font=('Arial', 12),
    width=20,
    textvariable=selected_item_var
)
label.pack()

# Function to handle Listbox item selection
def print_selection():
    try:
        # Get the selected item
        selected_index = item_listbox.curselection()
        if selected_index:  # Ensure something is selected
            selected_value = item_listbox.get(selected_index)
            selected_item_var.set(selected_value)  # Set the Label text
        else:
            selected_item_var.set("No selection")
    except Exception as e:
        selected_item_var.set(f"Error: {e}")

# Create a Button to trigger the selection print
print_button = tk.Button(
    window,
    text='Print Selection',
    width=15,
    height=2,
    command=print_selection
)
print_button.pack()

# String variable for the Listbox items (not directly used)
list_var = tk.StringVar()

# Create a Listbox and populate it with items
item_listbox = tk.Listbox(window)

# Add items to the Listbox
list_items = [11, 22, 33, 44]
for item in list_items:
    item_listbox.insert('end', item)
item_listbox.insert(1, 'first')  # Insert at index 1
item_listbox.insert(2, 'second')  # Insert at index 2
item_listbox.delete(2)  # Delete item at index 2
item_listbox.pack()

# Start the main event loop
window.mainloop()
