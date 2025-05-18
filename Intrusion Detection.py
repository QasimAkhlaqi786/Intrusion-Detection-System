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

# Hugging Face Transformers imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Colors and styles
BG_COLOR = "#2c3e50"
FG_COLOR = "#ecf0f1"
ACCENT_COLOR = "#3498db"
BUTTON_COLOR = "#2980b9"
TEXT_COLOR = "#2c3e50"
ERROR_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"

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

# --- Hugging Face DialoGPT local model loading and chat logic ---
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.eval()

chat_history_ids = None  # to keep conversation context

def local_chatbot_response(message):
    global chat_history_ids
    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

def hf_chatbot_response(message):
    return local_chatbot_response(message)

def create_placeholder(text):
    img = Image.new('RGB', (400, 400), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text(((400 - w) / 2, (400 - h) / 2), text, fill=FG_COLOR, font=font)
    return ImageTk.PhotoImage(img)

class App:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition Security System + AI Assistant")
        root.geometry("1000x700")
        root.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', background=BUTTON_COLOR, foreground=FG_COLOR, padding=10)
        style.map('TNotebook.Tab', background=[('selected', ACCENT_COLOR)])
        style.configure('TFrame', background=BG_COLOR)
        style.configure('Accent.TButton', background=BUTTON_COLOR, foreground=FG_COLOR)
        style.map('Accent.TButton', background=[('active', ACCENT_COLOR)])

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.face_frame = ttk.Frame(self.notebook)
        self.chat_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.face_frame, text="Face Recognition")
        self.notebook.add(self.chat_frame, text="AI Chatbot")

        self.setup_face_recognition()
        self.setup_chatbot()

    def setup_face_recognition(self):
        left_panel = ttk.Frame(self.face_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        self.input_panel = ttk.Label(left_panel, image=create_placeholder("No image loaded"))
        self.input_panel.pack(pady=10)

        btn_select = ttk.Button(left_panel, text="Select Image", command=self.select_image, style='Accent.TButton')
        btn_select.pack(fill=tk.X, pady=(10, 5))

        btn_check = ttk.Button(left_panel, text="Check Image", command=self.check_image, style='Accent.TButton')
        btn_check.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(left_panel, text="Load an image to start", foreground=FG_COLOR, background=BG_COLOR)
        self.status_label.pack(pady=(10, 0))

        right_panel = ttk.Frame(self.face_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.result_display = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=('Arial', 12), bg=FG_COLOR, fg=TEXT_COLOR)
        self.result_display.pack(fill=tk.BOTH, expand=True)
        self.result_display.config(state=tk.DISABLED)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
            title="Select an image file"
        )
        if file_path:
            try:
                img = Image.open(file_path).resize((400, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.input_panel.config(image=img_tk)
                self.input_panel.image = img_tk
                self.input_panel.image_path = file_path
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", foreground=SUCCESS_COLOR)
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}", foreground=ERROR_COLOR)

    def check_image(self):
        if hasattr(self.input_panel, "image_path"):
            try:
                result, sim, thresh, name, proc_time = recognize_face(self.input_panel.image_path)
                explanation = generate_explanation(result, sim, thresh, name, proc_time)

                self.result_display.config(state=tk.NORMAL)
                self.result_display.delete(1.0, tk.END)
                self.result_display.insert(tk.END, explanation)
                self.result_display.config(state=tk.DISABLED)

                self.status_label.config(text="Recognition complete", foreground=SUCCESS_COLOR)
            except Exception as e:
                self.status_label.config(text=f"Error during recognition: {str(e)}", foreground=ERROR_COLOR)
        else:
            self.status_label.config(text="No image loaded", foreground=ERROR_COLOR)

    def setup_chatbot(self):
        frame = ttk.Frame(self.chat_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=('Arial', 11), bg=FG_COLOR, fg=TEXT_COLOR)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.tag_config("user", foreground=ACCENT_COLOR)
        self.chat_display.tag_config("ai", foreground=SUCCESS_COLOR)

        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=10)

        self.input_entry = ttk.Entry(input_frame, font=('Arial', 11))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_entry.bind("<Return>", self.send_message)

        btn_send = ttk.Button(input_frame, text="Send", command=self.send_message, style='Accent.TButton')
        btn_send.pack(side=tk.RIGHT)

        self.add_chat_message("ai", "AI Assistant: How can I help you today?")

    def add_chat_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{'You' if sender == 'user' else 'AI'}: {message}\n\n", sender)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self, event=None):
        message = self.input_entry.get().strip()
        if not message:
            return

        self.input_entry.delete(0, tk.END)
        self.add_chat_message("user", message)

        response = hf_chatbot_response(message)
        self.add_chat_message("ai", response)

        if any(word in message.lower() for word in ["bye", "exit", "quit"]):
            self.add_chat_message("ai", "Goodbye! Closing chatbot soon...")
            self.root.after(2000, self.root.quit)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
