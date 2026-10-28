
import streamlit as st

st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.subheader("유튜브 영상 및 댓글 분석")

st.write(
    "유튜브 영상 링크를 입력하고 원하는 분석 기능을 선택해 보세요."
)

video_url = st.text_input(
    "유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.markdown("### 분석할 기능 선택")

video_stats = st.checkbox("영상 정보 분석", value=True)
comment_analysis = st.checkbox("댓글 감정 분석")
wordcloud_analysis = st.checkbox("워드클라우드")
popular_views = st.checkbox("채널 조회수 TOP 5")
popular_likes = st.checkbox("채널 좋아요 TOP 5")

if st.button("분석 시작", type="primary"):
    if not video_url.strip():
        st.warning("유튜브 영상 링크를 입력해 주세요.")
    else:
        st.info("웹사이트 기본 화면이 완성됐어요. 다음 단계에서 분석 기능을 연결할게요.")
