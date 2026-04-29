from pathlib import Path
# Load MathJax file at startup
print("=" * 60)
print("🔍 Checking for local MathJax file...")

MATHJAX_FILE = Path("tex-chtml.js")

if MATHJAX_FILE.exists():
    file_size = MATHJAX_FILE.stat().st_size
    print(f"✅ Found tex-chtml.js ({file_size / 1024:.1f} KB)")
    
    try:
        with open(MATHJAX_FILE, "r", encoding="utf-8") as f:
            mathjax_script_content = f.read()
        
        # Validate file content
        if mathjax_script_content.startswith('<!DOCTYPE') or mathjax_script_content.startswith('<html'):
            print("❌ ERROR: File contains HTML instead of JavaScript!")
            print("   Please re-download: curl -o tex-chtml.js https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js")
            mathjax_script_content = None
        elif not mathjax_script_content.startswith('(function'):
            print("⚠️ WARNING: File doesn't start with expected pattern")
            print(f"   First 50 chars: {mathjax_script_content[:50]}")
        else:
            print(f"✅ Successfully loaded {len(mathjax_script_content)} characters")
            print(f"   First 50 chars: {mathjax_script_content[:50]}...")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        mathjax_script_content = None
else:
    print(f"❌ File not found at: {MATHJAX_FILE.absolute()}")
    mathjax_script_content = None

print("=" * 60)

import gradio as gr
import pandas as pd
import random
import datetime
import os
import json
from datasets import load_dataset, Dataset
import time
import re
import queue
import threading
from pathlib import Path

# ==================== CONFIGURATION ====================
HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = os.environ.get("DATASET_REPO", "your-username/model-evaluation-votes")
QUESTIONS_FILE = "questions.csv"

MAX_PER_USER = 20
DEFAULT_MAX_EVALUATIONS = 20

CACHE_DIR = Path("vote_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ==================== QUEUE SYSTEM ====================
vote_queue = queue.Queue()
failed_votes = []
is_processing = False
processing_thread = None
lock = threading.Lock()


def push_to_hf_with_retry(votes_batch):
    # global append_count
    if not votes_batch:
        return True
    
    max_retries = 10
    base_delay = 2

    for attempt in range(max_retries):
        try:
            try:
                dataset = load_dataset(DATASET_REPO, split='train', token=HF_TOKEN)
                df_existing = dataset.to_pandas()
            except Exception as e:
                print(f"⚠️ Cannot load dataset, creating new: {e}")
                df_existing = pd.DataFrame()
            
            df_new = pd.concat([df_existing, pd.DataFrame(votes_batch)], ignore_index=True)
            dataset = Dataset.from_pandas(df_new)
            dataset.push_to_hub(DATASET_REPO, token=HF_TOKEN, commit_message=f"Add {len(votes_batch)} votes", max_shard_size="500MB")
            
            print(f"✅ Pushed {len(votes_batch)} votes to HF")
            return True
            
        except Exception as e:
            if "429" in str(e):
                delay = min(base_delay * (3 ** attempt), 350)
                print(f"⚠️ Rate limit, waiting {delay:.1f}s...")
                time.sleep(delay)
            elif attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 60)
                jitter = random.uniform(0, delay * 0.3)
                wait_time = delay + jitter
                print(f"⚠️ Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts: {e}")
                return False
    return False


def process_vote_queue():
    global is_processing, failed_votes
    is_processing = True
    print("🚀 Queue processor started")
    
    while is_processing:
        batch = []  # Initialize batch before try
        try:
            try:
                vote_data = vote_queue.get(timeout=180)
            except queue.Empty:
                with lock:
                    if failed_votes:
                        print(f"🔄 Retrying {len(failed_votes)} failed votes...")
                        if push_to_hf_with_retry(failed_votes):
                            failed_votes = []
                        time.sleep(5)
                continue
            
            if vote_data == "STOP":
                print("🛑 Stop signal received")
                break
            
            batch = [vote_data]
            try:
                while len(batch) < 100:
                    next_vote = vote_queue.get(timeout=60)
                    if next_vote == "STOP":
                        vote_queue.put("STOP")
                        break
                    batch.append(next_vote)
            except queue.Empty:
                pass

            
            print(f"📦 Processing batch of {len(batch)} votes...")
            
            if push_to_hf_with_retry(batch):
                for _ in batch:
                    try:
                        vote_queue.task_done()
                    except ValueError:
                        pass
            else:
                print(f"⚠️ Batch failed, adding to retry queue")

                with lock:
                    failed_votes.extend(batch)
                for _ in batch:
                    try:
                        vote_queue.task_done()
                    except ValueError:
                        pass
            
        except Exception as e:
            print(f"❌ Queue processor error: {e}")
            import traceback
            traceback.print_exc()
            # Safeguard: persist batch if crash occurs during processing
            if batch:
                print(f"🛟 Recovering {len(batch)} votes from crashed batch")
                with lock:
                    failed_votes.extend(batch)
            time.sleep(1)
    
    with lock:
        if failed_votes:
            print(f"🔄 Final attempt for {len(failed_votes)} failed votes...")
            push_to_hf_with_retry(failed_votes)
    
    print("✅ Queue processor stopped")

def start_queue_processor():
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_vote_queue, daemon=True)
        processing_thread.start()
        print("🚀 Vote queue processor started")

def stop_queue_processor():
    global is_processing
    print("🛑 Stopping queue processor...")
    is_processing = False
    vote_queue.put("STOP")
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=15)
    print("✅ Queue processor stopped")

def get_queue_stats():
    with lock:
        return {'queue_size': vote_queue.qsize(), 'failed_votes': len(failed_votes), 'is_processing': is_processing}

# ==================== FUNCTIONS ====================
def protect_latex_syntax(content):
    if not content:
        return ""
    content = str(content)
    
    # First, escape the delimiters (as before)
    content = re.sub(r'\\\(', r'\\\\(', content)
    content = re.sub(r'\\\)', r'\\\\)', content)
    content = re.sub(r'\\\[', r'\\\\[', content)
    content = re.sub(r'\\\]', r'\\\\]', content)
    
    # Expanded patterns to include $$ ... $$ and $ ... $
    latex_patterns = [
        r'\\\[([\s\S]*?)\\\]',      # Display: \[ ... \]
        r'\\\(([\s\S]*?)\\\)',      # Inline: \( ... \)
        r'\$\$([\s\S]*?)\$\$',      # Display: $$ ... $$
        r'\$([\s\S]*?)\$',          # Inline: $ ... $ (non-greedy)
    ]
    
    for pattern in latex_patterns:
        def replacer(match):
            inner = match.group(1)
            # Trim leading/trailing newlines (and optionally other whitespace)
            trimmed_inner = inner.strip('\n')  # Or .strip() for all whitespace
            # Protect '|' inside the LaTeX (as before)
            protected_inner = trimmed_inner.replace('|', r'\|')
            # Reconstruct the full match with original delimiters
            return match.group(0).replace(inner, protected_inner, 1)
        
        content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    return content

def format_content(content):
    if not content:
        return ""
    return protect_latex_syntax(str(content))

def load_questions():
    try:
        if not os.path.exists(QUESTIONS_FILE):
            raise FileNotFoundError(f"File {QUESTIONS_FILE} không tồn tại.")
        
        df = pd.read_csv(QUESTIONS_FILE, encoding='utf-8')
        required_cols = ['question_id', 'category', 'content_turn1', 'response_turn1_a']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV thiếu các cột: {missing_cols}")
        
        if 'max_evaluations' not in df.columns:
            df['max_evaluations'] = DEFAULT_MAX_EVALUATIONS
        
        text_cols = ['content_turn1', 'response_turn1_a', 'response_turn1_b', 
                     'content_turn2', 'response_turn2_a', 'response_turn2_b',
                     'reference_turn1', 'reference_turn2']
        
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        return df
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        raise

def load_votes():
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset = load_dataset(DATASET_REPO, split='train', token=HF_TOKEN)
            df = dataset.to_pandas()
            # Remove fully duplicate rows across all columns
            df = df.drop_duplicates(keep='first')
            return df.copy()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"⚠️ Cannot load votes: {e}")
                return pd.DataFrame(columns=['timestamp', 'question_id', 'category', 'model_a', 'model_b',
                    'conversation_a', 'conversation_b', 'judge_name', 'judge_type',
                    'judgement_explanation', 'verdict', 'judge_email'])
            time.sleep(0.5 * (attempt + 1))


data_lock = threading.Lock()
df_votes = load_votes()

def log_vote(question_row, verdict, judge_name, explanation="", is_single=False, judge_email=""):
    timestamp = datetime.datetime.now().isoformat()
    
    try:
        conversation_a = [
            {"content": str(question_row['content_turn1']), "role": "user"},
            {"content": str(question_row['response_turn1_a']), "role": "assistant"}
        ]
        
        conversation_b = []
        if not is_single and question_row.get('response_turn1_b'):
            conversation_b = [
                {"content": str(question_row['content_turn1']), "role": "user"},
                {"content": str(question_row['response_turn1_b']), "role": "assistant"}
            ]
        
        judge_type = "single_turn"
        if question_row.get('content_turn2') and question_row.get('response_turn2_a'):
            conversation_a.extend([
                {"content": str(question_row['content_turn2']), "role": "user"},
                {"content": str(question_row['response_turn2_a']), "role": "assistant"}
            ])
            if not is_single and question_row.get('response_turn2_b'):
                conversation_b.extend([
                    {"content": str(question_row['content_turn2']), "role": "user"},
                    {"content": str(question_row['response_turn2_b']), "role": "assistant"}
                ])
            judge_type = "multi_turn"
        
        if question_row.get('reference_turn1'):
            judge_type += "_with_reference"
        
        judge_type += "_single" if is_single else "_comparison"
        
        new_row = {
            'timestamp': timestamp,
            'question_id': question_row['question_id'],
            'category': question_row['category'],
            'model_a': question_row.get('model_a', 'N/A'),
            'model_b': question_row.get('model_b', 'N/A'),
            'conversation_a': json.dumps(conversation_a, ensure_ascii=False),
            'conversation_b': json.dumps(conversation_b, ensure_ascii=False) if conversation_b else "",
            'judge_name': judge_name,
            'judge_type': judge_type,
            'judgement_explanation': explanation,
            'verdict': verdict,
            'judge_email': judge_email
        }

        # Critical: update votes in df_votes
        with data_lock:
            global df_votes
            new_df_row = pd.DataFrame([new_row])  # Convert dict to single-row DataFrame
            df_votes = pd.concat([df_votes, new_df_row], ignore_index=True)
        
        vote_queue.put(new_row)
        queue_size = vote_queue.qsize()
        
        with lock:
            failed_count = len(failed_votes)
        
        print(f"📥 Vote queued (queue: {queue_size}, failed: {failed_count})")
        
        if queue_size > 50:
            print(f"⚠️ Queue is long ({queue_size}), processing may be slow")
        
        return True
    except Exception as e:
        print(f"❌ Error queuing vote: {e}")
        import traceback
        traceback.print_exc()
        return False

def select_question(df_questions, user_done):
    global df_votes
    try:
        with data_lock:  # Add lock to protect df_votes read operation
            vote_counts = df_votes.groupby('question_id').size().reset_index(name='vote_count')
        merged = df_questions.merge(vote_counts, on='question_id', how='left').fillna({'vote_count': 0})
        available = merged[(~merged['question_id'].isin(user_done)) & (merged['vote_count'] < merged['max_evaluations'])]
        
        if available.empty:
            return None
        
        available = available.sort_values('vote_count', ascending=True).reset_index(drop=True)
        
        # Randomly select from the 5 lowest-voted items
        top_n = min(5, len(available))
        selected = available.head(top_n).sample(1)
        
        return selected.iloc[0].to_dict()
    except Exception as e:
        return None


def validate_email(email):
    if not email or not email.strip():
        return False, "Email không được để trống"
    
    email = email.strip()
    if '@' not in email or '.' not in email:
        return False, "Email phải có dạng: example@domain (vd domain: gmail.com; student.hcmus.edu.vn; ...)"
    
    parts = email.split('@')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, "Email phải có domain hợp lệ (vd: gmail.com; student.hcmus.edu.vn; ...)"
    
    if '.' not in parts[1]:
        return False, "Email phải có domain hợp lệ (vd: gmail.com; student.hcmus.edu.vn; ...)"
    
    return True, ""

# ==================== GRADIO UI ====================
custom_css = """
.question-box,
.response-box,
.reference-box {
    padding-left: 0.2em !important;
    min-width: 50%;
    min-height: 100px !important;
    will-change: transform;
}
/* RESET: Ngăn scroll lồng nhau từ gốc */
.response-box table, .reference-box table {
    max-width: 100%;
    display: block;
    overflow-x: auto;
    -webkit-overflow-scroll: touch;
}
.question-box {
    border-radius: 8px;
    margin: 10px 0;
    border: 2px dashed #FF9800;
    position: relative;
    max-height: 40vh;
}
.question-label {
    font-weight: bold; margin-bottom: 8px; color: var(--text-color-primary); font-size: 1.1em;
}
/* FIX CHÍNH: Chỉ MỘT container scroll */
.response-box {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    background: var(--background-fill-primary);
    max-height: 40vh;
    line-height: 1.5;
    font-size: 0.95em;
   
    /* QUAN TRỌNG: Chỉ có box cha được scroll */
    overflow-y: auto;
    overflow-x: auto;
}
.gr-table {
    width: 100% !important;
    min-width: min-content !important;
    table-layout: auto !important;
}
/* Cho tất cả bảng trong app */
.markdown table, .prose table {
    width: 100% !important;
    min-width: min-content !important;
    table-layout: auto !important;
    display: table !important;
}
/* Các ô trong bảng */
.markdown td, .markdown th,
.prose td, .prose th {
    white-space: nowrap !important;
    min-width: 150px !important;
}
/* Container cho bảng */
.markdown, .prose {
    overflow-x: auto !important;
    max-width: 100% !important;
}
.reference-box {
    border-radius: 12px;
    border-left: 5px solid #4CAF50;
    color: white; 
    font-size: 1.1rem; 
    font-weight: 500; 
    line-height: 1.6; 
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    /* max-height: 40vh; */  /* Comment out or remove this */
        /* 🚨 QUAN TRỌNG: Chỉ MỘT scroll container */
    /*overflow: auto !important;*/
    /* 🚨 THÊM 2 DÒNG NÀY */
    overflow-y: auto !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;  /* Momentum scrolling cho iOS */
    scrollbar-width: thin;  /* Áp dụng scrollbar styling tương tự */
    scrollbar-color: #FF9800 transparent;  /* Màu phù hợp với border dashed */
}
.math-display {
    text-align: center;
    margin: 10px 0;
    overflow-x: auto;
    padding: 10px 0;
}
.email-container {
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.email-container label {
    color: white !important;
    font-size: 1.1em;
    font-weight: 600;
}
.rating-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 0.85em;
    color: var(--text-color-secondary);
}
/* ===== SCROLLBAR STYLING ===== */
/* Chỉ áp dụng cho .response-box */
.response-box {
    scrollbar-width: thin;
    scrollbar-color: #4CAF50 transparent;
}
.response-box::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
.response-box::-webkit-scrollbar-track {
    background: transparent;
    margin: 5px 0;
}
.response-box::-webkit-scrollbar-thumb {
    background: linear-gradient(to bottom, #4CAF50, #2E7D32);
    border-radius: 5px;
    border: 2px solid transparent;
    background-clip: content-box;
}
.response-box::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(to bottom, #66BB6A, #388E3C);
}


.reference-box::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

.reference-box::-webkit-scrollbar-track {
    background: transparent;
    margin: 5px 0;
}

.reference-box::-webkit-scrollbar-thumb {
    background: linear-gradient(to bottom, #FF9800, #F57C00);
    border-radius: 5px;
    border: 2px solid transparent;
    background-clip: content-box;
}

.reference-box::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(to bottom, #FFB74D, #EF6C00);
}

.button-row {
    display: flex;
    flex-direction: row; /* Mặc định ngang trên desktop */
    justify-content: space-between; /* Khoảng cách đều */
    align-items: center;
    gap: 10px; /* Khoảng cách giữa buttons */
}
/* ===== MOBILE FIXES ===== */
@media (max-width: 768px) {
    .button-row {
        flex-direction: column !important; /* Ép dọc trên mobile */
        align-items: stretch; /* Buttons full width */
        gap: 15px; /* Tăng khoảng cách dọc cho đẹp */
    }

    .button-row > .gr-button { /* Target trực tiếp buttons con */
        width: 100%; /* Full width trên mobile */
        margin: 0 !important; /* Xóa margin thừa */
    }
    /* QUAN TRỌNG: Force chỉ có một scroll container */
    body, html {
        overflow-x: auto;
    }
   
    .response-box {
        max-height: 250px;
        overflow-y: auto !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }
   
    /* Đảm bảo không có nested scrollbars */
    .response-box .markdown {
        display: block;
        overflow: auto !important;
       
    }
   
    /* Scrollbar mobile - rõ ràng hơn */
    .response-box::-webkit-scrollbar {
        width: 12px;
    }
   
    .reference-box {
        overflow: auto;
       
    }
   
    .markdown {
        margin: 8px 0 !important;
    }
    .reference-box {
        max-height: 250px;  /* Đồng bộ với .response-box để tránh lệch */
        overflow-y: auto !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    
    /* Ngăn nested scroll triệt để hơn cho element con */
    .reference-box .markdown,
    .reference-box .prose,
    .reference-box table {
        overflow: visible !important;  /* Không cho con scroll riêng, đẩy hết lên cha */
        max-width: 100%;
    }
    
    /* Sử dụng % thay vh cho height dynamic hơn */
    .reference-box {
        max-height: 40%;  /* Hoặc calc(40vh - 50px) để trừ address bar */
    }
}
@media (max-width: 480px) {
    .question-box {
        font-size: 1rem;
    }
   
    .response-box {
        font-size: 0.9em;
        max-height: 200px;
        will-change: transform;
    }
    .reference-box {
        max-height: 200px;
        will-change: transform;
    }
}
@media (min-width: 769px) {
    .question-box {
        font-size: 1.25rem;
        will-change: transform;
    }
   
    .response-box {
        font-size: 1em;
        max-height: 350px;
        will-change: transform;
    }
    .reference-box {  /* Add this */
        max-height: 350px;  /* Sync with .response-box */
        /* Ensure overflow is enabled if not inherited */
        overflow-y: auto !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }
}
"""
mathjax_config = """
<script>
window.MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true },
    chtml: {
        // ⚡ QUAN TRỌNG: Chỉ định CDN path cho fonts
        fontURL: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/output/chtml/fonts/woff-v2'
    }
};
</script>
<script >""" + mathjax_script_content+"""</script>
"""

with gr.Blocks(title="Model Evaluator", css=custom_css) as demo:
    gr.HTML(mathjax_config)
    
    #session_id = gr.State(value=lambda: str(random.randint(100000, 999999)))
    session_id = gr.State("")
    user_vote_count = gr.State(0)
    user_done_questions = gr.State([])
    current_question = gr.State()
    is_single_mode = gr.State(False)
    user_email = gr.State("")
    
    mathjax_trigger = gr.HTML(value="", elem_id="mathjax-trigger", visible=True)

    gr.Markdown("""
    # 🎯 Đánh giá mô hình
    """, elem_id="main-title")

    with gr.Column(elem_classes="email-container") as email_section:
        gr.Markdown("### 📧 Thông tin người đánh giá")
        email_input = gr.Textbox(label="Email của bạn (bắt buộc)", placeholder="example@domain", 
                                 info="Email sẽ được lưu cùng đánh giá của bạn")
        email_error = gr.Markdown(visible=False)
        email_submit = gr.Button("✅ Bắt đầu đánh giá", variant="primary", size="lg")

    try:
        df_questions = load_questions()
    except Exception as e:
        gr.Error(f"Không thể load: {e}")
        df_questions = pd.DataFrame()

    status = gr.Markdown("⏳ Vui lòng nhập email để bắt đầu...", visible=False)
    
    comparison_container = gr.Column(visible=False)
    with comparison_container:
        # gr.Markdown("### Turn 1")
        # gr.Markdown("**Question**", elem_classes="question-label")
        question_turn1 = gr.Markdown(elem_classes="question-box")
        reference_turn1 = gr.Markdown(visible=False, elem_classes="reference-box")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Phản hồi của mô hình A")
                response_a_turn1 = gr.Markdown(elem_classes="response-box")
            with gr.Column():
                gr.Markdown("### Phản hồi của mô hình B")
                response_b_turn1 = gr.Markdown(elem_classes="response-box")
        
        turn2_container = gr.Column(visible=False)
        with turn2_container:
            # gr.Markdown("### Turn 2")
            # gr.Markdown("**Follow-up Question**", elem_classes="question-label")
            question_turn2 = gr.Markdown(elem_classes="question-box")
            reference_turn2 = gr.Markdown(visible=False, elem_classes="reference-box")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Phản hồi của mô hình A ")
                    response_a_turn2 = gr.Markdown(elem_classes="response-box")
                with gr.Column():
                    gr.Markdown("### Phản hồi của mô hình B")
                    response_b_turn2 = gr.Markdown(elem_classes="response-box")
        
        explanation_comp = gr.Textbox(label="💭 Giải thích cho lựa chọn (Không bắt buộc)", 
                                     placeholder="Giải thích lý do đánh giá của bạn...", lines=2, max_lines=4)
        
        with gr.Row(elem_classes="button-row"):
            btn_a = gr.Button("👈 Mô hình A tốt hơn", variant="primary", size="lg")
            btn_tie = gr.Button("🤝 Hòa", variant="secondary", size="lg")
            btn_b = gr.Button("👉 Mô hình B tốt hơn", variant="primary", size="lg")
    
    single_container = gr.Column(visible=False)
    with single_container:
        # gr.Markdown("### Turn 1")
        # gr.Markdown("**Question**", elem_classes="question-label")
        question_turn1_single = gr.Markdown(elem_classes="question-box")
        reference_turn1_single = gr.Markdown(visible=False, elem_classes="reference-box")
        gr.Markdown("### Phản hồi của mô hình")
        response_single_turn1 = gr.Markdown(elem_classes="response-box")
        
        turn2_container_single = gr.Column(visible=False)
        with turn2_container_single:
            # gr.Markdown("### Turn 2")
            # gr.Markdown("**Follow-up Question**", elem_classes="question-label")
            question_turn2_single = gr.Markdown(elem_classes="question-box")
            reference_turn2_single = gr.Markdown(visible=False, elem_classes="reference-box")
            gr.Markdown("### Phản hồi của mô hình")
            response_single_turn2 = gr.Markdown(elem_classes="response-box")
        
        explanation_single = gr.Textbox(label="💭 Giải thích cho lựa chọn (Không bắt buộc)", 
                                       placeholder="Giải thích lý do đánh giá của bạn...", lines=2, max_lines=4)

        rating_slider = gr.Slider(minimum=0, maximum=10, step=1, value=5, label="⭐ Đánh giá (0-10)"
                                 )
        
        gr.HTML("""<div class="rating-labels">
            <span>😞 Không hài lòng (0)</span><span>😐 Tạm ổn (5)</span><span>😊 Rất hài lòng (10)</span></div>""")

        btn_submit = gr.Button("✅ Gửi đánh giá", variant="primary", size="lg")

    # ==================== EDIT initial_load ====================
    def initial_load(done, count, sid, email):
        """Khởi tạo ban đầu - chỉ hiển thị form email nếu chưa có email"""
        # If email exists, load questions; otherwise, display email form
        if email and email.strip() and '@' in email and '.' in email:
            # Valid email provided -> load questions
            return load_question(done, count, sid, email)
        else:
            # No email provided -> display email form
            timestamp = time.time()
            trigger_html = f'<div style="display:none" data-mathjax-ready="{timestamp}">SIGNAL-{timestamp}</div>'
            return (
                trigger_html,
                None, False, "⏳ Vui lòng nhập email để bắt đầu đánh giá",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", ""
            )
    
    # ==================== EDIT HÀM check_email ====================
    def check_email(email):
        """Kiểm tra email và trả về kết quả"""
        valid, msg = validate_email(email)
        if valid:
            ses_id = str(random.randint(100000, 999999))
            # Valid email -> hide email form and display status
            return (
                email.strip(),  # user_email state
                ses_id,
                gr.update(visible=False),  # hide email_section
                gr.update(visible=False),  # hide email_error
                gr.update(visible=True, value="✅ Email hợp lệ! Đang tải câu hỏi...")  # status
            )
        else:
            # Invalid email -> display error message
            return (
                "",  # Do not update email (keep empty)
                "",
                gr.update(visible=True),  # keep email_section visible
                gr.update(visible=True, value=f"❌ {msg}"),  # display error
                gr.update(visible=False)  # hide status
            )
    
    # ==================== EDIT email_submit.click ====================
    # Only call check_email and load_question
    email_submit.click(
        check_email,
        inputs=[email_input],
        outputs=[user_email, session_id, email_section, email_error, status]
    ).then(
        lambda done, count, sid, email: load_question(done, count, sid, email),
        inputs=[user_done_questions, user_vote_count, session_id, user_email],
        outputs=[
            mathjax_trigger, current_question, is_single_mode, status,
            comparison_container, question_turn1, reference_turn1, reference_turn1, response_a_turn1, response_b_turn1,
            turn2_container, question_turn2, reference_turn2, reference_turn2, response_a_turn2, response_b_turn2,
            single_container, question_turn1_single, reference_turn1_single, reference_turn1_single, response_single_turn1,
            turn2_container_single, question_turn2_single, reference_turn2_single, reference_turn2_single, response_single_turn2
        ]
    )

    def load_question(user_done, count, sess_id, email):
        timestamp = time.time()
        trigger_html = f'<div style="display:none" data-mathjax-ready="{timestamp}">SIGNAL-{timestamp}</div>'
        print(f"🔔 Python sending signal: {timestamp}")

        if not email or not isinstance(email, str) or email.strip() == "":
            print(f"❌ Email chưa được nhập hoặc rỗng")
            return (
                trigger_html,
                None, False, "❌ Vui lòng nhập email hợp lệ để bắt đầu đánh giá",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", ""
            )
        
        try:

            question_row = select_question(df_questions, user_done)
            
            if question_row is None:
                return (trigger_html, None, False, f"🎉 Không còn câu hỏi! Bạn đã hoàn thành {count} câu đánh giá.",
                    gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                    gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                    gr.update(visible=False), "", gr.update(visible=False), "", "",
                    gr.update(visible=False), "", gr.update(visible=False), "", "")
            
            is_single = not question_row.get('model_b') or not question_row.get('response_turn1_b')
            status_text ="Đánh giá chất lượng câu trả lời của các mô hình AI.\n\n💡 **Lưu ý:** Vui lòng giải thích lý do đánh giá. Nếu lỗi, tải lại trang - dữ liệu đã được lưu.\n"
            status_text += f"\n## 📝 Câu hỏi thứ **{count + 1}** | **Loại:** {question_row['category']}"
            status_text += (
                " | ⭐ **Phương thức:** Đánh giá đơn | Chọn đánh giá 0 (Không hài lòng) → 10 (Rất hài lòng)"
                if is_single
                else " | 🆚 **Phương thức:** So sánh | Chọn mô hình A tốt hơn, B tốt hơn, hoặc hòa nhau"
            )
            
            q1 = format_content(f"### Câu hỏi lượt 1:\n\n{question_row['content_turn1']}")
            ref1_text = format_content(f"### 💡 Câu trả lời mẫu:\n\n{question_row['reference_turn1']}") if question_row.get('reference_turn1') else ""
            ref1_vis = gr.update(visible=bool(question_row.get('reference_turn1')))
            
            has_turn2 = bool(question_row.get('content_turn2') and question_row.get('response_turn2_a'))
            q2 = format_content(f"### Câu hỏi lượt 2:\n\n{question_row.get('content_turn2', '')}") if has_turn2 else ""
            ref2_text = format_content(f"### 💡 Câu trả lời mẫu:\n\n{question_row['reference_turn2']}") if has_turn2 and question_row.get('reference_turn2') else ""
            ref2_vis = gr.update(visible=(has_turn2 and bool(question_row.get('reference_turn2'))))
            
            if is_single:
                a1 = format_content(f"{question_row['response_turn1_a']}")
                a2 = format_content(f"{question_row.get('response_turn2_a', '')}") if has_turn2 else ""
                return (trigger_html, question_row, True, status_text,
                    gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                    gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                    gr.update(visible=True), q1, ref1_vis, ref1_text, a1,
                    gr.update(visible=has_turn2), q2, ref2_vis, ref2_text, a2)
            else:
                a1 = format_content(f"{question_row['response_turn1_a']}")
                b1 = format_content(f"{question_row['response_turn1_b']}")
                a2 = format_content(f"{question_row.get('response_turn2_a', '')}") if has_turn2 else ""
                b2 = format_content(f"{question_row.get('response_turn2_b', '')}") if has_turn2 else ""
                return (trigger_html, question_row, False, status_text,
                    gr.update(visible=True), q1, ref1_vis, ref1_text, a1, b1,
                    gr.update(visible=has_turn2), q2, ref2_vis, ref2_text, a2, b2,
                    gr.update(visible=False), "", gr.update(visible=False), "", "",
                    gr.update(visible=False), "", gr.update(visible=False), "", "")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
            return (trigger_html, None, False, f"⚠️ Lỗi không lưu đánh giá",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "",
                gr.update(visible=False), "", gr.update(visible=False), "", "")
    
    def vote_comp(verdict, q, done, count, sid, expl, email):
        if q is None:
            return done, count, *load_question(done, count, sid, email), ""
        if log_vote(q, verdict, sid, expl, False, email):
            new_done = done + [q['question_id']]
            return new_done, count + 1, *load_question(new_done, count + 1, sid, email), ""
        else:
            return done, count, f'<div data-mathjax-ready="{time.time()}"></div>', q, False, "⚠️ Lỗi không lưu đánh giá", *[gr.update() for _ in range(22)], ""

    def vote_sing(rating, q, done, count, sid, expl, email):
        if q is None:
            return done, count, *load_question(done, count, sid, email), 5, ""
        if log_vote(q, str(rating), sid, expl, True, email):
            new_done = done + [q['question_id']]
            return new_done, count + 1, *load_question(new_done, count + 1, sid, email), 5, ""
        else:
            return done, count, f'<div data-mathjax-ready="{time.time()}"></div>', q, True, "⚠️ Lỗi không lưu đánh giá", *[gr.update() for _ in range(22)], rating, ""

    for btn, v in [(btn_a, "A"), (btn_tie, "C"), (btn_b, "B")]:
        btn.click(vote_comp,
            [gr.State(v), current_question, user_done_questions, user_vote_count, session_id, explanation_comp, user_email],
            [user_done_questions, user_vote_count, mathjax_trigger, current_question, is_single_mode, status,
             comparison_container, question_turn1, reference_turn1, reference_turn1, response_a_turn1, response_b_turn1,
             turn2_container, question_turn2, reference_turn2, reference_turn2, response_a_turn2, response_b_turn2,
             single_container, question_turn1_single, reference_turn1_single, reference_turn1_single, response_single_turn1,
             turn2_container_single, question_turn2_single, reference_turn2_single, reference_turn2_single, response_single_turn2,
             explanation_comp])
    
    btn_submit.click(vote_sing,
        [rating_slider, current_question, user_done_questions, user_vote_count, session_id, explanation_single, user_email],
        [user_done_questions, user_vote_count, mathjax_trigger, current_question, is_single_mode, status,
         comparison_container, question_turn1, reference_turn1, reference_turn1, response_a_turn1, response_b_turn1,
         turn2_container, question_turn2, reference_turn2, reference_turn2, response_a_turn2, response_b_turn2,
         single_container, question_turn1_single, reference_turn1_single, reference_turn1_single, response_single_turn1,
         turn2_container_single, question_turn2_single, reference_turn2_single, reference_turn2_single, response_single_turn2,
         rating_slider, explanation_single])
    
    gr.HTML("""
    <script>
    console.log('🚀 Script loaded');
    (function() {
        'use strict';
        let rendering = false;

        function scrollAllBoxesToTop() {
            const boxes = document.querySelectorAll('.question-box, .response-box, .reference-box');
            boxes.forEach(box => {
                if (box) {
                    box.scrollTop = 0;
                    console.log('📜 Scrolled box to top:', box.className);
                }
            });
        }
        
        function renderMathJax(reason) {
            console.log(`🔄 renderMathJax called (${reason})`);
            if (rendering) {
                console.log('⏳ Already rendering, skip');
                return;
            }
            if (!window.MathJax) {
                console.error('❌ MathJax not found!');
                return;
            }
            if (!window.MathJax.typesetPromise) {
                console.error('❌ MathJax.typesetPromise not found!');
                return;
            }
            rendering = true;
            console.log('✅ Calling MathJax.typesetPromise()...');
            MathJax.typesetPromise()
                .then(() => {
                    console.log('✅ MathJax rendered successfully!');
                    rendering = false;
                })
                .catch(e => {
                    console.error('❌ MathJax error:', e);
                    rendering = false;
                });
        }
        
        function watchTrigger() {
            const el = document.querySelector('#mathjax-trigger');
            if (!el) {
                console.log('⏳ Trigger not found, retrying...');
                setTimeout(watchTrigger, 200);
                return;
            }
            console.log('✅ Found #mathjax-trigger:', el);
            console.log('Current content:', el.innerHTML);
            
            const observer = new MutationObserver(muts => {
                console.log('🔔 MutationObserver fired!', muts.length, 'mutations');
                for (let m of muts) {
                    console.log('  Type:', m.type, 'AddedNodes:', m.addedNodes.length);
                    if (m.type === 'childList' && m.addedNodes.length > 0) {
                        for (let node of m.addedNodes) {
                            console.log('  Node:', node.nodeType, node.nodeName, node);
                            if (node.nodeType === 1) {
                                const attr = node.getAttribute('data-mathjax-ready');
                                console.log('  data-mathjax-ready:', attr);
                                if (attr) {
                                    console.log(`📢 SIGNAL RECEIVED: ${attr}`);
                                    
                                    // ✅ CÁCH 1: Scroll tới tiêu đề chính
                                    setTimeout(() => {
                                        // 1️⃣ Scroll tất cả boxes về đầu
                                        scrollAllBoxesToTop();

                                        /*
                                        // Tìm phần tử tiêu đề chính bằng ID
                                        const mainTitle = document.querySelector('#main-title');
                                        
                                        if (mainTitle) {
                                            
                                            // Code cũ bị comment ra
                                            mainTitle.scrollIntoView({ 
                                                behavior: 'smooth', 
                                                block: 'start', 
                                                inline: 'nearest' 
                                            });
                                            console.log('✅ Scrolled to main title');
                                            
                            
                                        } else {
                                            // Fallback: scroll to top
                                            window.scrollTo({
                                                top: 0,
                                                behavior: 'smooth'
                                            });
                                        }
                                        */
                                        const mainTitle = document.querySelector('#main-title');
                                        if (mainTitle) {
                                            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
    
                                            if (isIOS) {
                                                // ✅ Tạo input tạm
                                                const tempInput = document.createElement('input');
                                                tempInput.style.position = 'absolute';
                                                tempInput.style.opacity = '0'; // Hide
                                                tempInput.style.height = '1px'; // Small size
                                                mainTitle.parentNode.insertBefore(tempInput, mainTitle); // Insert trước mainTitle
                                                
                                                tempInput.focus(); // Trigger scroll parent
                                                
                                                setTimeout(() => {
                                                    tempInput.blur();
                                                    tempInput.remove(); // Cleanup
                                                }, 500);
                                                
                                                console.log('✅ iOS temp input focus');
                                            } else {
                                                // Android: Bình thường
                                                mainTitle.scrollIntoView({ 
                                                    behavior: 'smooth', 
                                                    block: 'start',
                                                    inline: 'nearest'
                                                });
                                                console.log('✅ Android scrollIntoView');
                                            }
                                        }
                                        
                                        
                                        // Render MathJax sau khi scroll
                                        setTimeout(() => renderMathJax('signal'), 150); 
                                    }, 300);
                                    

                                    return;
                                }
                            }
                        }
                    }
                }
            });
            
            observer.observe(el, {
                childList: true,
                subtree: true,
                attributes: false,
                characterData: false
            });
            console.log('👁️ Observer watching #mathjax-trigger');
        }
        
        function init() {
            console.log('🎬 Initializing...');
            if (window.MathJax && window.MathJax.startup) {
                console.log('✅ MathJax found');
                MathJax.startup.promise.then(() => {
                    console.log('✅ MathJax ready');
                    watchTrigger();
                    setTimeout(() => renderMathJax('init'), 300);
                });
            } else {
                console.log('⏳ MathJax not ready, retrying...');
                setTimeout(init, 200);
            }
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
        
        window.__renderMathJax = () => renderMathJax('manual');
        window.__checkTrigger = () => {
            const el = document.querySelector('#mathjax-trigger');
            console.log('Trigger element:', el);
            console.log('Trigger HTML:', el ? el.innerHTML : 'NOT FOUND');
        };
    })();
    </script>
    """)

if __name__ == "__main__":
    import atexit
    start_queue_processor()
    
    def cleanup():
        print("\n🧹 Cleaning up...")
        stop_queue_processor()
        if not vote_queue.empty():
            print(f"⏳ Waiting for {vote_queue.qsize()} remaining votes...")
            try:
                vote_queue.join()
            except:
                pass
        stats = get_queue_stats()
        print(f"📊 Final stats: {stats}")
        print("✅ Cleanup done!")
    
    atexit.register(cleanup)
    demo.launch()