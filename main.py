import os
import asyncio
from dotenv import load_dotenv
from langchain.prompts import load_prompt, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
import pdfplumber
import pandas as pd
import shutil

load_dotenv()  # .env 파일에서 환경 변수 로드

# OpenAI API 키 설정 (환경 변수에 OPENAI_API_KEY로 저장되어 있어야 함)
# 예: os.environ["OPENAI_API_KEY"] = 'your_api_key'

# LLM 설정
llm = ChatOpenAI(temperature=0, model_name="gpt-4")  # 모델명은 사용 가능한 것으로 설정

# 프롬프트 로드
template = load_prompt("translation_prompt.yaml", encoding="UTF-8")

# 프롬프트 템플릿 설정
prompt = PromptTemplate(template=template.template, input_variables=["text"])

# 번역 체인 생성 (RunnableSequence 사용)
translation_chain = prompt | llm

# 경로 설정
pdf_folder = "pdf"
translated_folder = "temp_translated"
final_folder = "final_translated"

# 로그 파일 경로
translation_log_file = "translation_log.txt"
final_log_file = "final_log.txt"


# 로그 불러오기 함수
def load_log(log_file):
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.read().splitlines()
    else:
        return []


translation_processed_files = load_log(translation_log_file)
final_processed_files = load_log(final_log_file)

# PDF 파일 목록 가져오기
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]


# 콘텐츠 추출 함수 (이미지 및 테이블 제외)
def extract_content_from_page(pdf_path, page_number):
    content = ""
    # 텍스트 추출
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        text = page.extract_text() or ""
        content += text + "\n"

    return content


# 페이지 번역 함수
async def translate_page(pdf_name, page_number, pdf_path):
    output_dir = os.path.join(translated_folder, pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{page_number}.md"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"{pdf_name}/page{page_number} 이미 번역됨. 건너뜁니다.")
        return

    content = extract_content_from_page(pdf_path, page_number)

    try:
        # 번역 실행
        result = await translation_chain.ainvoke({"text": content})
        translated_text = result.content
    except Exception as e:
        print(f"번역 실패: {pdf_name}/page{page_number}, 에러: {str(e)}")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated_text)

    print(f"{pdf_name}/page{page_number} 번역 완료.")


# PDF 처리 함수
async def process_pdf(pdf_file):
    pdf_name = os.path.splitext(pdf_file)[0]
    if pdf_name in translation_processed_files:
        print(f"{pdf_name} 이미 번역됨. 건너뜁니다.")
        return

    pdf_path = os.path.join(pdf_folder, pdf_file)
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)

    tasks = []
    for page_number in range(1, num_pages + 1):
        tasks.append(translate_page(pdf_name, page_number, pdf_path))

    await asyncio.gather(*tasks)

    with open(translation_log_file, "a") as f:
        f.write(f"{pdf_name}\n")

    print(f"{pdf_name} 번역 완료.")


# 번역된 파일 합치기 함수
def merge_translations(pdf_name):
    if pdf_name in final_processed_files:
        print(f"{pdf_name} 이미 합쳐짐. 건너뜁니다.")
        return

    translated_dir = os.path.join(translated_folder, pdf_name)
    output_path = os.path.join(final_folder, f"{pdf_name}.md")

    if not os.path.exists(translated_dir):
        print(f"{pdf_name} 번역된 파일이 없습니다.")
        return

    page_files = [f for f in os.listdir(translated_dir) if f.endswith(".md")]
    page_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    combined_text = ""
    for page_file in page_files:
        page_path = os.path.join(translated_dir, page_file)
        with open(page_path, "r", encoding="utf-8") as f:
            page_content = f.read()
            combined_text += page_content + "\n\n"

    os.makedirs(final_folder, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    with open(final_log_file, "a") as f:
        f.write(f"{pdf_name}\n")

    print(f"{pdf_name} 합치기 완료.")


# 메인 함수
async def main():
    tasks = []
    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(pdf_file)[0]
        if pdf_name in final_processed_files:
            print(f"{pdf_name} 이미 완료됨. 건너뜁니다.")
            continue
        tasks.append(process_pdf(pdf_file))
    await asyncio.gather(*tasks)

    for pdf_file in pdf_files:
        pdf_name = os.path.splitext(pdf_file)[0]
        if pdf_name in final_processed_files:
            continue
        merge_translations(pdf_name)


# 실행
if __name__ == "__main__":
    asyncio.run(main())
