import tkinter
import tkinter.messagebox
 
def buttonClick():
    tkinter.messagebox.showinfo('title', 'message')
    #tkinter.messagebox.showwarning('title', 'message')
    #tkinter.messagebox.showerror('title', 'message')
 
root=tkinter.Tk()
root.title('GUI')  
root.geometry('100x100')  
root.resizable(False, False)  
tkinter.Button(root, text='hello button',command=buttonClick).pack()
root.mainloop()
