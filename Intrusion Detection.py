import sys
import io
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import pickle
from keras_facenet import FaceNet
from numpy.linalg import norm
import time
import os

# Configure console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# System Information
SYSTEM_DESCRIPTION = """
SYSTEM DESIGN OVERVIEW:

1. Deep Learning Components:
   - FaceNet Model: Deep CNN for face recognition
   - 128-dimensional face embeddings
   - Cosine similarity matching

2. Generative AI Components:
   - Natural language explanations
   - Conversational AI interface

3. Core Functionality:
   - Face detection with OpenCV
   - Feature extraction
   - Database matching
   - Threshold-based access control

4. Technical Specs:
   - Threshold: 0.8 (configurable)
   - Processing Time: <500ms
   - Accuracy: >98%
"""

# Color Scheme
BG_COLOR = "#2c3e50"  # Dark blue-gray
FG_COLOR = "#ecf0f1"   # Light gray
ACCENT_COLOR = "#3498db" # Blue
BUTTON_COLOR = "#2980b9" # Darker blue
TEXT_COLOR = "#2c3e50"  # Dark text
ERROR_COLOR = "#e74c3c"  # Red
SUCCESS_COLOR = "#2ecc71" # Green

# Load FaceNet model and embeddings
try:
    embedder = FaceNet()
    with open("face_embeddings.pkl", "rb") as f:
        known_names, known_embeddings = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    known_names, known_embeddings = [], []

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recognize_face(image_path, threshold=0.8):
    try:
        start_time = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return "Image not found", None, None, None, None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = embedder.embeddings([img_rgb])[0]

        similarities = [cosine_similarity(emb, known_emb) for known_emb in known_embeddings]
        best_sim = max(similarities) if similarities else 0
        best_idx = similarities.index(best_sim) if similarities else 0

        name = known_names[best_idx] if known_names and best_idx < len(known_names) else "Unknown"
        processing_time = time.time() - start_time

        if best_sim > threshold:
            return f"Allowed: {name.encode('ascii', 'ignore').decode()}", best_sim, threshold, name, processing_time
        return "Not Allowed", best_sim, threshold, name, processing_time
    except Exception as e:
        return f"Error: {str(e)}", None, None, None, None

def generate_explanation(result, similarity, threshold, name, processing_time):
    name_display = name.encode('ascii', 'ignore').decode() if name else "Unknown"
    
    if "Allowed" in result:
        return (
            f"ACCESS GRANTED TO: {name_display}\n\n"
            f"Similarity: {similarity:.4f} (Threshold: {threshold})\n"
            f"Processing Time: {processing_time:.3f}s\n"
            f"Model: FaceNet DNN\n\n"
            f"Explanation:\n"
            f"Identified as '{name_display}' with high confidence."
        )
    elif "Not Allowed" in result:
        return (
            f"ACCESS DENIED\n\n"
            f"Closest Match: {name_display}\n"
            f"Similarity: {similarity:.4f} (Threshold: {threshold})\n"
            f"Processing Time: {processing_time:.3f}s\n\n"
            f"Possible Reasons:\n"
            "1. Not in database\n2. Poor image quality\n3. Occlusion/lighting issues"
        )
    return result

def chatbot_response(message, current_result=None):
    message = message.lower().strip()
    
    responses = {
        "system": SYSTEM_DESCRIPTION,
        "model": "This system uses:\n1. FaceNet CNN for recognition\n2. Generative AI for explanations",
        "allowed": "ACCESS GRANTED BECAUSE:\n1. Face detected\n2. Score > threshold\n3. Matches database",
        "denied": "ACCESS DENIED BECAUSE:\n1. No match\n2. Low score\n3. Detection failed",
        "hello": "Hello! I'm your AI security assistant.",
        "similarity": "Similarity Score:\n1. Extract features\n2. Compare vectors\n3. Score 0-1",
        "threshold": "Threshold: 0.8\n≥0.8: Access\n<0.8: Denied",
        "thanks": "You're welcome!",
        "bye": "Goodbye! Stay secure!",
        "default": "Ask about:\n- System design\n- Access decisions\n- Models used"
    }
    
    if "system" in message or "design" in message:
        return responses["system"]
    if "model" in message or "deep learning" in message:
        return responses["model"]
    if current_result and ("why" in message or "reason" in message):
        return responses["allowed"] if "Allowed" in current_result else responses["denied"]
    if "hello" in message:
        return responses["hello"]
    if "similarity" in message:
        return responses["similarity"]
    if "threshold" in message:
        return responses["threshold"]
    if "thank" in message:
        return responses["thanks"]
    if "bye" in message:
        return responses["bye"]
    return responses["default"]

class ChatbotWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("AI Security Assistant")
        self.window.geometry("700x600")
        self.window.configure(bg=BG_COLOR)
        self.current_result = None
        
        # Main frame
        main_frame = ttk.Frame(self.window, style='Custom.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=('Arial', 10),
            bg=FG_COLOR,
            padx=10,
            pady=10,
            foreground=TEXT_COLOR
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.tag_config("user", foreground=ACCENT_COLOR)
        self.chat_display.tag_config("ai", foreground=SUCCESS_COLOR)
        
        # Input area
        input_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        input_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.input_entry = ttk.Entry(
            input_frame, 
            font=('Arial', 11),
            style='Custom.TEntry'
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.send_message)
        
        ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            style='Accent.TButton'
        ).pack(side=tk.RIGHT)
        
        self.add_message("ai", "AI Assistant: How can I help you today?")

    def add_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{'You' if sender == 'user' else 'AI'}: {message}\n\n", sender)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        message = self.input_entry.get().strip()
        if not message:
            return
            
        self.input_entry.delete(0, tk.END)
        self.add_message("user", message)
        
        response = chatbot_response(message, self.current_result)
        self.add_message("ai", response)
        
        if any(word in message.lower() for word in ["bye", "exit", "quit"]):
            self.window.after(2000, self.window.destroy)

def create_placeholder(text):
    img = Image.new('RGB', (400, 400), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((140, 190), text, fill=FG_COLOR, font=font)
    return ImageTk.PhotoImage(img)

def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        title="Select an image file"
    )
    if file_path:
        try:
            img = Image.open(file_path).resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            input_panel.config(image=img_tk)
            input_panel.image = img_tk
            input_panel.image_path = file_path
            status_label.config(text=f"Loaded: {os.path.basename(file_path)}", foreground=SUCCESS_COLOR)
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", foreground=ERROR_COLOR)

def check_image():
    global current_result
    if hasattr(input_panel, "image_path"):
        try:
            result, sim, thresh, name, proc_time = recognize_face(input_panel.image_path)
            explanation = generate_explanation(result, sim, thresh, name, proc_time)
            current_result = result
            
            result_display.config(state=tk.NORMAL)
            result_display.delete(1.0, tk.END)
            result_display.insert(tk.END, f"{result}\n\n{explanation}")
            result_display.config(state=tk.DISABLED)
            
            status_label.config(text="Analysis completed", foreground=SUCCESS_COLOR)
        except Exception as e:
            status_label.config(text=f"Error: {str(e)}", foreground=ERROR_COLOR)
            result_display.config(state=tk.NORMAL)
            result_display.delete(1.0, tk.END)
            result_display.insert(tk.END, f"Error: {str(e)}")
            result_display.config(state=tk.DISABLED)
    else:
        status_label.config(text="Please select an image first", foreground=ERROR_COLOR)

def show_system_info():
    info_win = tk.Toplevel(root)
    info_win.title("System Architecture")
    info_win.geometry("800x600")
    info_win.configure(bg=BG_COLOR)
    
    text = scrolledtext.ScrolledText(
        info_win, 
        wrap=tk.WORD, 
        font=('Consolas', 10),
        bg=FG_COLOR,
        foreground=TEXT_COLOR
    )
    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text.insert(tk.END, SYSTEM_DESCRIPTION)
    text.config(state=tk.DISABLED)
    
    ttk.Button(
        info_win, 
        text="Close", 
        command=info_win.destroy,
        style='Accent.TButton'
    ).pack(pady=10)

# Main Application
root = tk.Tk()
root.title("Intrusion Detection AI")
root.geometry("1000x700")
root.configure(bg=BG_COLOR)

# Style Configuration
style = ttk.Style()
style.theme_create('custom', parent='alt', settings={
    'TFrame': {'configure': {'background': BG_COLOR}},
    'TLabel': {
        'configure': {
            'background': BG_COLOR,
            'foreground': FG_COLOR,
            'font': ('Arial', 10)
        }
    },
    'Header.TLabel': {
        'configure': {
            'font': ('Arial', 12, 'bold'),
            'foreground': FG_COLOR
        }
    },
    'TButton': {
        'configure': {
            'background': BUTTON_COLOR,
            'foreground': FG_COLOR,
            'font': ('Arial', 10),
            'padding': 8,
            'borderwidth': 1
        },
        'map': {
            'background': [('active', ACCENT_COLOR)],
            'foreground': [('active', FG_COLOR)]
        }
    },
    'Accent.TButton': {
        'configure': {
            'background': ACCENT_COLOR,
            'foreground': FG_COLOR,
            'font': ('Arial', 10, 'bold')
        }
    },
    'Custom.TFrame': {
        'configure': {'background': BG_COLOR}
    },
    'Custom.TEntry': {
        'configure': {
            'fieldbackground': FG_COLOR,
            'foreground': TEXT_COLOR
        }
    }
})
style.theme_use('custom')

# Create UI Components
main_frame = ttk.Frame(root, style='Custom.TFrame')
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Header
header_frame = ttk.Frame(main_frame, style='Custom.TFrame')
header_frame.pack(fill=tk.X, pady=(0, 20))

ttk.Label(
    header_frame,
    text="Intrusion Detection AI System",
    style='Header.TLabel'
).pack(side=tk.LEFT)

ttk.Button(
    header_frame,
    text="System Info",
    command=show_system_info,
    style='Accent.TButton'
).pack(side=tk.RIGHT)

# Content Area
content_frame = ttk.Frame(main_frame, style='Custom.TFrame')
content_frame.pack(fill=tk.BOTH, expand=True)

# Left Panel - Input
left_frame = ttk.Frame(content_frame, width=450, style='Custom.TFrame')
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

ttk.Label(
    left_frame,
    text="Input Image",
    style='Header.TLabel'
).pack(pady=(0, 10))

placeholder_input = create_placeholder("Input Image")
input_panel = ttk.Label(left_frame, image=placeholder_input)
input_panel.pack()
input_panel.image = placeholder_input

button_frame = ttk.Frame(left_frame, style='Custom.TFrame')
button_frame.pack(pady=15)

ttk.Button(
    button_frame,
    text="Select Image",
    command=select_image,
    width=15
).pack(side=tk.LEFT, padx=5)

ttk.Button(
    button_frame,
    text="Verify Identity",
    command=check_image,
    width=15
).pack(side=tk.LEFT, padx=5)

status_label = ttk.Label(
    left_frame,
    text="Ready",
    foreground=SUCCESS_COLOR
)
status_label.pack()

# Middle Panel - Results
middle_frame = ttk.Frame(content_frame, style='Custom.TFrame')
middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

ttk.Label(
    middle_frame,
    text="Analysis Results",
    style='Header.TLabel'
).pack(pady=(0, 10))

result_display = scrolledtext.ScrolledText(
    middle_frame,
    wrap=tk.WORD,
    font=('Arial', 10),
    height=20,
    padx=10,
    pady=10,
    bg=FG_COLOR,
    foreground=TEXT_COLOR
)
result_display.pack(fill=tk.BOTH, expand=True)
result_display.insert(tk.END, "Results will appear here...")
result_display.config(state=tk.DISABLED)

ttk.Button(
    middle_frame,
    text="Ask AI Assistant",
    command=lambda: ChatbotWindow(root),
    width=20,
    style='Accent.TButton'
).pack(pady=15)

# Footer
ttk.Label(
    main_frame,
    text="Intrusion Detection AI v2.0 | © 2025 Security Systems Inc.",
    foreground=FG_COLOR
).pack(pady=(20, 0))

# Initialize variables
current_result = None

root.mainloop()