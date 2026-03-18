
# 여기에 GenAI가 생성한 코드를 붙여넣으세요
import streamlit as st
import cv2
import numpy as np

# --- 필터 처리 함수 모음 (모든 입출력은 BGR 이미지 기준) ---
def apply_original(img):
    return img.copy()

def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 비교를 위해 3채널 유지

def apply_gaussian(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_canny(img, t1, t2):
    edges = cv2.Canny(img, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_sepia(img):
    # 정석적인 세피아 톤 적용을 위해 BGR -> RGB -> 변환 -> BGR 순서로 처리
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    sepia = cv2.transform(img_rgb, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)

def apply_sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1,  5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# --- 앱 UI 구성 ---
st.set_page_config(page_title="이미지 필터 비교 앱", layout="wide")
st.title("🖼️ 이미지 필터 비교 앱")

# 사이드바: 필터 설정
st.sidebar.title("⚙️ 필터 설정")
filter_choice = st.sidebar.selectbox(
    "필터 선택", 
    ["원본", "회색조", "Gaussian 블러", "Canny 엣지", "세피아", "선명화"]
)

# 필터별 추가 설정 (슬라이더)
ksize = 15
t1, t2 = 100, 200

if filter_choice == "Gaussian 블러":
    ksize = st.sidebar.slider("블러 커널 크기 (홀수)", min_value=1, max_value=31, value=15, step=2)
elif filter_choice == "Canny 엣지":
    t1 = st.sidebar.slider("Threshold 1", min_value=0, max_value=255, value=100)
    t2 = st.sidebar.slider("Threshold 2", min_value=0, max_value=255, value=200)

# 메인 화면: 이미지 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 업로드된 이미지를 메모리에서 읽어 OpenCV 포맷(BGR)으로 변환
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. 선택한 필터 적용
    if filter_choice == "원본":
        result_bgr = apply_original(img_bgr)
    elif filter_choice == "회색조":
        result_bgr = apply_grayscale(img_bgr)
    elif filter_choice == "Gaussian 블러":
        result_bgr = apply_gaussian(img_bgr, ksize)
    elif filter_choice == "Canny 엣지":
        result_bgr = apply_canny(img_bgr, t1, t2)
    elif filter_choice == "세피아":
        result_bgr = apply_sepia(img_bgr)
    elif filter_choice == "선명화":
        result_bgr = apply_sharpen(img_bgr)

    # 3. Streamlit 출력을 위해 BGR을 RGB로 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # 4. 화면 분할 및 결과 출력
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(img_rgb, use_container_width=True)
    with col2:
        st.subheader(f"필터 적용: {filter_choice}")
        st.image(result_rgb, use_container_width=True)

    # 5. 결과 이미지 다운로드 버튼 구현
    # OpenCV 이미지를 PNG 포맷 바이트 배열로 인코딩
    is_success, buffer = cv2.imencode(".png", result_bgr)
    if is_success:
        st.download_button(
            label="📥 필터 적용 이미지 다운로드 (PNG)",
            data=buffer.tobytes(),
            file_name="filtered_image.png",
            mime="image/png"
        )
