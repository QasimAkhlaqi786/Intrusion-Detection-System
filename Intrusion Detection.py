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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure console output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==================== CONSTANTS ====================
BG_COLOR = "#2c3e50"
FG_COLOR = "#ecf0f1"
ACCENT_COLOR = "#3498db"
BUTTON_COLOR = "#2980b9"
TEXT_COLOR = "#2c3e50"
ERROR_COLOR = "#e74c3c"
SUCCESS_COLOR = "#2ecc71"

# ==================== FACE RECOGNITION ====================
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

# ==================== TECHNICAL ASSISTANT ====================
class FaceRecognitionAssistant:
    def __init__(self):
        self.system_info = {
            "name": "FRAS Helper",
            "version": "1.2",
            "model": "FaceNet DNN",
            "threshold": 0.8,
            "features": {
                "recognition": "Real-time face detection and identification",
                "security": "Access control based on facial similarity",
                "logging": "Detailed recognition events recording"
            }
        }
        self.technical_manual = {
            "output_format": {
                "allowed": "Format: 'Allowed: [Name]', Confidence: [0-1], Processing Time: [seconds]",
                "denied": "Format: 'Not Allowed', Closest Match: [Name], Confidence: [0-1]",
                "error": "Format: 'Error: [description]'"
            },
            "process_flow": [
                "1. Image captured/loaded",
                "2. Face detection using MTCNN",
                "3. Feature extraction with FaceNet",
                "4. Database comparison (cosine similarity)",
                "5. Threshold comparison",
                "6. Output generation"
            ],
            "troubleshooting": {
                "low_confidence": [
                    "Check lighting conditions",
                    "Ensure face is clearly visible",
                    "Verify person is in database"
                ],
                "false_positives": [
                    "Adjust threshold (current: 0.8)",
                    "Add more reference images",
                    "Check for similar-looking individuals in DB"
                ]
            }
        }

    def explain_output(self, output):
        if "Allowed:" in output:
            return (
                f"ACCESS GRANTED EXPLANATION:\n"
                f"This means the system recognized a face from its database with high confidence.\n"
                f"Technical Details:\n"
                f"- Matching Process: {self.technical_manual['process_flow'][3]}\n"
                f"- Threshold Comparison: Current threshold = {self.system_info['threshold']}\n"
                f"- Typical Confidence Range: 0.85-0.99 for reliable matches"
            )
        elif "Not Allowed" in output:
            return (
                f"ACCESS DENIED EXPLANATION:\n"
                f"The system didn't find a match above the confidence threshold.\n"
                f"Possible Reasons:\n"
                + "\n".join(f"- {item}" for item in self.technical_manual['troubleshooting']['low_confidence'])
            )
        elif "Error:" in output:
            return (
                f"ERROR ANALYSIS:\n"
                f"This indicates a system or input problem.\n"
                f"Common Fixes:\n"
                f"- Verify image file integrity\n"
                f"- Check system logs for details\n"
                f"- Ensure model files are properly loaded"
            )
        return "This output format isn't recognized. Please provide a standard system output."

    def explain_process(self, process_name):
        process_name = process_name.lower()
        if "detect" in process_name:
            return (
                "FACE DETECTION PROCESS:\n"
                "Uses MTCNN (Multi-task Cascaded Convolutional Networks) to:\n"
                "- Identify face locations in images\n"
                "- Detect facial landmarks\n"
                "- Handle multiple faces in single image"
            )
        elif "extract" in process_name or "feature" in process_name:
            return (
                "FEATURE EXTRACTION:\n"
                "FaceNet model converts faces to 128-dimension embeddings:\n"
                "- Captures unique facial characteristics\n"
                "- Creates compact representation for comparison\n"
                "- Normalized for cosine similarity calculations"
            )
        elif "compare" in process_name or "match" in process_name:
            return (
                "DATABASE COMPARISON:\n"
                "System performs these steps:\n"
                "1. Calculates cosine similarity between input and database embeddings\n"
                "2. Scores range from 0 (no similarity) to 1 (identical)\n"
                f"3. Compares against threshold ({self.system_info['threshold']})\n"
                "4. Returns best match if above threshold"
            )
        return f"I can explain: detection, extraction, or comparison processes. Which one?"

    def get_technical_answer(self, question):
        question = question.lower()
        
        if "how work" in question or "how it work" in question:
            return "FACE RECOGNITION PROCESS:\n" + "\n".join(self.technical_manual['process_flow'])
        
        elif "threshold" in question:
            return (
                f"THRESHOLD EXPLANATION (Current: {self.system_info['threshold']}):\n"
                "This confidence score determines access:\n"
                "- < threshold: Access denied\n"
                "- ≥ threshold: Access granted\n"
                "Adjust in config.py if needed"
            )
            
        elif "improve" in question or "accuracy" in question:
            return (
                "IMPROVING ACCURACY:\n"
                "1. Add more reference images per person\n"
                "2. Ensure consistent lighting in captures\n"
                "3. Update embeddings periodically\n"
                f"4. Adjust threshold (current: {self.system_info['threshold']})"
            )
            
        elif "database" in question or "storage" in question:
            return (
                "FACE DATABASE INFO:\n"
                "- Stores 128D face embeddings\n"
                "- Associated with person names\n"
                "- Saved in face_embeddings.pkl\n"
                "- Add new faces via enroll.py"
            )
            
        return None

fras_assistant = FaceRecognitionAssistant()

# ==================== CHATBOT ====================
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
model.eval()
chat_history = None

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

def chatbot_response(message):
    # First check for technical questions
    tech_answer = fras_assistant.get_technical_answer(message)
    if tech_answer:
        return tech_answer
        
    # Handle output explanations
    if "explain output" in message.lower() or "what does this mean" in message.lower():
        sample_output = "Allowed: User (Confidence: 0.92, Time: 0.45s)"
        return fras_assistant.explain_output(sample_output)
        
    # Handle process explanations
    if "explain" in message.lower() and ("process" in message.lower() or "step" in message.lower()):
        return fras_assistant.explain_process(message)
        
    # Default to general chatbot
    global chat_history
    new_input = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    
    if chat_history is not None:
        bot_input = torch.cat([chat_history, new_input], dim=-1)[:,-512:]
    else:
        bot_input = new_input
        
    chat_history = model.generate(
        bot_input,
        max_length=512,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    
    response = tokenizer.decode(chat_history[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    return response

# ==================== GUI ====================
class App:
    def __init__(self, root):
        self.root = root
        root.title("Face Recognition Security System + Technical Assistant")
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
        self.notebook.add(self.chat_frame, text="Technical Assistant")

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

        btn_explain = ttk.Button(left_panel, text="Explain Result", 
                               command=self.explain_result, 
                               style='Accent.TButton')
        btn_explain.pack(fill=tk.X, pady=5)

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
                output = f"{result}\nSimilarity: {sim:.4f}\nThreshold: {thresh}\nProcessing Time: {proc_time:.3f}s"
                
                self.result_display.config(state=tk.NORMAL)
                self.result_display.delete(1.0, tk.END)
                self.result_display.insert(tk.END, output)
                self.result_display.config(state=tk.DISABLED)
                
                self.input_panel.last_result = output  # Store for explanation
                self.status_label.config(text="Recognition complete", foreground=SUCCESS_COLOR)
            except Exception as e:
                self.status_label.config(text=f"Error during recognition: {str(e)}", foreground=ERROR_COLOR)
        else:
            self.status_label.config(text="No image loaded", foreground=ERROR_COLOR)

    def explain_result(self):
        if hasattr(self.input_panel, "last_result"):
            explanation = fras_assistant.explain_output(self.input_panel.last_result)
            self.result_display.config(state=tk.NORMAL)
            self.result_display.insert(tk.END, "\n\n=== EXPLANATION ===\n" + explanation)
            self.result_display.config(state=tk.DISABLED)
        else:
            self.status_label.config(text="No result to explain", foreground=ERROR_COLOR)

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

        btn_reset = ttk.Button(input_frame, text="Reset", command=self.reset_conversation, style='Accent.TButton')
        btn_reset.pack(side=tk.RIGHT, padx=5)

        self.add_chat_message("ai", "Technical Assistant: Ask me about the face recognition system or its outputs.")

    def add_chat_message(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{'You' if sender == 'user' else 'Assistant'}: {message}\n\n", sender)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def send_message(self, event=None):
        message = self.input_entry.get().strip()
        if not message:
            return

        self.input_entry.delete(0, tk.END)
        self.add_chat_message("user", message)

        response = chatbot_response(message)
        self.add_chat_message("ai", response)

    def reset_conversation(self):
        global chat_history
        chat_history = None
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.add_chat_message("ai", "Conversation reset. Ask me about the face recognition system.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
