from tkinter import *

root = Tk()

textLabel = Label(root,
                  text="Label",
                  justify=LEFT,
                  padx=10)
textLabel.pack(side=LEFT)

photo = PhotoImage(file="cat.png")
imgLabel = Label(root, image=photo)
imgLabel.pack(side=RIGHT)

mainloop()
