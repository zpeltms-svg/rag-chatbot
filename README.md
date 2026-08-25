# 화성시 민원 챗봇

`data/` 폴더의 TXT 문서를 OpenAI 임베딩으로 벡터화하여 ChromaDB에 저장하고, 질문과 유사한 문서 발췌만 근거로 답변하는 Streamlit 기반 RAG 챗봇입니다.

## 기능

- `data/*.txt` 문서를 문단 단위 청크로 분할해 ChromaDB에 영구 저장
- `text-embedding-3-small` 임베딩으로 유사 문서 검색
- 관련 문서가 없으면 `자료에서 확인할 수 없습니다`로 답변
- 답변과 함께 출처 파일명 및 검색된 발췌 표시
- 사이드바 버튼으로 데이터 변경 후 벡터 DB 재생성

## 설치

Python 3.10 이상을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## API 키 설정

`.env.example`을 복사해 `.env` 파일을 만들고 `OPENAI_API_KEY` 환경 변수를 설정합니다. 실제 키 파일은 Git에 포함되지 않습니다.

```powershell
Copy-Item .env.example .env
```

`.env` 파일 예시:

```env
OPENAI_API_KEY=your_api_key_here
```

## 실행

```powershell
streamlit run app.py
```

첫 실행에서 문서를 임베딩해 `chroma_db/`에 저장합니다. `data/` 문서를 추가하거나 수정한 경우 화면 왼쪽의 **벡터 DB 다시 만들기** 버튼을 누르세요.

## 동작 방식

1. TXT 문서를 청크로 나누고 OpenAI 임베딩을 생성합니다.
2. 임베딩과 메타데이터(파일명)를 ChromaDB에 저장합니다.
3. 질문 임베딩으로 상위 유사 문서를 검색합니다.
4. 유사도 기준을 통과한 발췌만 모델에 전달해 답변을 생성합니다.
5. 충분히 관련된 발췌가 없으면 정해진 안내 문구를 반환합니다.

모델 및 API 사용법은 [official OpenAI documentation](https://platform.openai.com/docs/quickstart/make-your-first-api-request)을 따릅니다.
