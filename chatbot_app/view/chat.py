from tkinter import *
from controller.botController import answer_question

BG_COLOR = "#17202A"        
TEXT_COLOR = "#EAECEE"     
BG_ENTRY = "#2C3E50"  
FG_USER = "#3498DB" 
FG_BOT = "#EAECEE" 
FONT = "Helvetica 12"
FONT_BOLD = "Helvetica 13 bold"

root = Tk()
root.title("Review Chatbot")
root.geometry("450x550")
root.configure(bg=BG_COLOR)

def send(event=None):
    msg = e.get()
    if not msg:
        return
    
    e.delete(0, END)
    
    txt.insert(END, "Você:\n", 'user_label')
    txt.insert(END, f"{msg}\n\n", 'user_msg')
    
    response = answer_question(msg)
    
    txt.insert(END, "Bot:\n", 'bot_label')
    txt.insert(END, f"{response}\n\n", 'bot_msg')
    
    txt.see(END)


header_label = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="Review Chatbot", 
                     font=FONT_BOLD, pady=15)
header_label.pack(fill=X)

chat_frame = Frame(root, bg=BG_COLOR)
chat_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)

scrollbar = Scrollbar(chat_frame)
scrollbar.pack(side=RIGHT, fill=Y)

txt = Text(chat_frame, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60, 
           bd=0, highlightthickness=0, yscrollcommand=scrollbar.set, wrap=WORD)
txt.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.config(command=txt.yview)

txt.tag_config('user_label', foreground=FG_USER, justify=RIGHT, font=("Helvetica", 10, "bold"))
txt.tag_config('user_msg', foreground=TEXT_COLOR, justify=RIGHT)
txt.tag_config('bot_label', foreground="#2ECC71", justify=LEFT, font=("Helvetica", 10, "bold"))
txt.tag_config('bot_msg', foreground=TEXT_COLOR, justify=LEFT)

bottom_frame = Frame(root, bg=BG_COLOR)
bottom_frame.pack(fill=X, padx=10, pady=15)

e = Entry(bottom_frame, bg=BG_ENTRY, fg=TEXT_COLOR, font=FONT, insertbackground='white', bd=0)
e.pack(side=LEFT, fill=X, expand=True, ipady=8, padx=(0, 10))
e.bind("<Return>", send)

send_btn = Button(bottom_frame, text="Enviar", font=("Helvetica", 10, "bold"), 
                  bg="#3498DB", fg="white", activebackground="#2980B9", 
                  activeforeground="white", bd=0, padx=20, command=send)
send_btn.pack(side=RIGHT)
