# main.py - 문제 잠금, CAPTCHA 인증, 요청 제한, 피드백 저장 포함

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import sqlite3
import pandas as pd
import os
import time
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.extension import Limiter
import httpx
from dotenv import load_dotenv

# 환경변수 로드 (CAPTCHA 비밀키 포함)
load_dotenv()
RECAPTCHA_SECRET = os.getenv("RECAPTCHA_SECRET")

# Google reCAPTCHA 검증 함수
def verify_recaptcha_sync(token: str) -> bool:
    url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": RECAPTCHA_SECRET, "response": token}
    try:
        response = httpx.post(url, data=data, timeout=5.0)
        result = response.json()
        return result.get("success", False)
    except Exception:
        return False

# FastAPI 인스턴스 (문서 비공개 설정)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# 사용자 식별 키 (user_id + IP)
def user_id_ip_key(request: Request):
    user_id = request.headers.get("X-User-Id", "")
    ip = get_remote_address(request)
    return f"{user_id}:{ip}"

app.state.limiter = Limiter(key_func=user_id_ip_key)

# 반복 제한 위반 카운터 (3회 → 블랙리스트)
violation_counter = {}

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    user_id = request.headers.get("X-User-Id", "")
    ip = get_remote_address(request)
    key = f"{user_id}:{ip}"
    violation_counter[key] = violation_counter.get(key, 0) + 1

    # 3회 초과 시 10분 차단
    if violation_counter[key] >= 3:
        now = int(time.time())
        block_until = now + 600
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO blacklist (user_id, block_until) VALUES (?, ?)", (user_id, block_until))
        conn.commit()
        conn.close()
        violation_counter[key] = 0
        return JSONResponse(
            status_code=429,
            content={"message": "과도한 요청"},
            headers={"X-Block-Remaining": "600"}
        )

    return JSONResponse(
        status_code=429,
        content={"detail": f"요청이 너무 많습니다. (1분 20회 제한, 누적 {violation_counter[key]}/3)"}
    )

# CORS 설정 (프론트 주소 등록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voluble-peony-178c73.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Block-Remaining"]
)

# 경로 및 DB 설정
DATA_ROOT = "./rag_quiz"
DB_PATH = "./quiz.db"
LOCK_TIMEOUT_SECONDS = 600  # 문제 잠금 시간

# DB 초기화 (문제, 피드백, 블랙리스트 테이블)
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS problems (...)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS feedbacks (...)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS blacklist (...)""")
    conn.commit()
    conn.close()

init_db()

# 요청용 모델 정의
class SubjectRequest(BaseModel):
    subject: str
    recaptcha_token: str

class ModelSelection(BaseModel):
    session_id: str
    subject: str
    idx: int
    selected_model: str

class FeedbackSubmission(BaseModel):
    session_id: str
    feedback: str

# NaN 방지 문자열 처리
def safe_text(val):
    return "" if pd.isna(val) else str(val)

# 블랙리스트 여부 확인
def is_user_blocked(user_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = int(time.time())
    cur.execute("SELECT block_until FROM blacklist WHERE user_id=? AND block_until > ?", (user_id, now))
    row = cur.fetchone()
    conn.close()
    return bool(row)

# 문제 비교 요청 처리
@app.post("/compare_models/")
def compare_models(req: SubjectRequest):
    if req.recaptcha_token:
        if not verify_recaptcha_sync(req.recaptcha_token):
            raise HTTPException(status_code=400, detail="CAPTCHA 인증 실패")

    subject = req.subject
    path_a = os.path.join(DATA_ROOT, subject, f"{subject}_modelA.csv")
    path_b = os.path.join(DATA_ROOT, subject, f"{subject}_modelB.csv")

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        raise HTTPException(status_code=404, detail="CSV 파일을 찾을 수 없습니다.")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now_ts = int(time.time())

    # 잠금 해제 (만료된 문제)
    cur.execute("DELETE FROM problems WHERE locked_until IS NOT NULL AND locked_until < ?", (now_ts,))
    conn.commit()

    selected_idx = None
    session_id = str(uuid4())

    # 잠기지 않은 문제 하나 선택
    for idx in range(len(df_a)):
        cur.execute("SELECT locked_until FROM problems WHERE subject=? AND idx=?", (subject, idx))
        row = cur.fetchone()
        if not row or (row[0] and row[0] < now_ts):
            try:
                cur.execute("INSERT OR REPLACE INTO problems (...) VALUES (...)",
                            (subject, idx, session_id, now_ts + LOCK_TIMEOUT_SECONDS))
                conn.commit()
                selected_idx = idx
                break
            except:
                conn.rollback()
                continue
    conn.close()

    if selected_idx is None:
        raise HTTPException(status_code=404, detail="모든 문제가 잠겨 있습니다.")

    q_a = df_a.iloc[selected_idx]
    q_b = df_b.iloc[selected_idx]

    return {
        "session_id": session_id,
        "idx": int(selected_idx),
        "model_a": {
            "question": safe_text(q_a.get("질문")),
            "choices": [safe_text(q_a.get(f"보기{i+1}")) for i in range(5)],
            "answer": safe_text(q_a.get("정답")),
            "explanation": safe_text(q_a.get("해설"))
        },
        "model_b": {
            "question": safe_text(q_b.get("질문")),
            "choices": [safe_text(q_b.get(f"보기{i+1}")) for i in range(5)],
            "answer": safe_text(q_b.get("정답")),
            "explanation": safe_text(q_b.get("해설"))
        }
    }

# 사용자 선택 저장 (rate limit: 1분당 20회)
@app.post("/save_selection/")
@app.state.limiter.limit("20/minute")
def save_selection(req: ModelSelection, request: Request):
    user_id = request.headers.get("X-User-Id", "")
    if is_user_blocked(user_id):
        raise HTTPException(status_code=429, detail="반복된 요청으로 10분 차단됨")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 권한 확인
    cur.execute("SELECT session_id FROM problems WHERE subject=? AND idx=?", (req.subject, req.idx))
    row = cur.fetchone()
    if not row or row[0] != req.session_id:
        raise HTTPException(status_code=403, detail="문제에 대한 권한 없음")

    # 잠금 해제 후 선택 저장
    cur.execute("UPDATE problems SET locked_until=NULL WHERE subject=? AND idx=?", (req.subject, req.idx))
    timestamp = int(time.time())
    cur.execute("""INSERT OR REPLACE INTO feedbacks (...) VALUES (...)""",
                (req.session_id, req.subject, req.selected_model, timestamp))
    conn.commit()
    conn.close()
    return {"status": "saved"}

# 피드백 저장
@app.post("/submit_feedback/")
def submit_feedback(req: FeedbackSubmission):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE feedbacks SET feedback=? WHERE session_id=?", (req.feedback, req.session_id))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    conn.commit()
    conn.close()
    return {"status": "feedback saved"}
