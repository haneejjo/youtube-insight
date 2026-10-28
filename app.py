
import os
import re
import tempfile
from io import BytesIO
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
from faster_whisper import WhisperModel


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 디자인 CSS - 3D 입체 버튼
# =========================================================

st.markdown("""
<style>
/* 페이지 전체의 기본 간격 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* 모든 버튼과 다운로드 버튼 */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 12px !important;
    border: 1px solid rgba(120, 120, 120, 0.25) !important;
    box-shadow:
        0 5px 0 rgba(70, 70, 70, 0.30),
        0 7px 12px rgba(0, 0, 0, 0.10) !important;
    transform: translateY(0);
    transition:
        transform 0.12s ease,
        box-shadow 0.12s ease,
        filter 0.12s ease !important;
    font-weight: 600 !important;
}

/* 마우스를 올리면 살짝 떠오름 */
.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 7px 0 rgba(70, 70, 70, 0.25),
        0 10px 16px rgba(0, 0, 0, 0.13) !important;
    filter: brightness(1.04);
}

/* 버튼을 누르면 아래로 들어가는 효과 */
.stButton > button:active,
.stDownloadButton > button:active,
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(4px);
    box-shadow:
        0 1px 0 rgba(70, 70, 70, 0.25),
        0 2px 5px rgba(0, 0, 0, 0.08) !important;
    filter: brightness(0.97);
}

/* 키보드로 버튼을 선택했을 때 */
.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 3px solid rgba(70, 130, 255, 0.65) !important;
    outline-offset: 3px;
}

/* 분석 시작 버튼을 조금 더 강조 */
.stButton > button[kind="primary"] {
    border-radius: 14px !important;
    box-shadow:
        0 6px 0 rgba(25, 75, 150, 0.65),
        0 9px 16px rgba(0, 0, 0, 0.12) !important;
    font-weight: 700 !important;
}

/* 분석 시작 버튼의 마우스 오버 */
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow:
        0 8px 0 rgba(25, 75, 150, 0.65),
        0 12px 18px rgba(0, 0, 0, 0.15) !important;
}

/* 분석 시작 버튼을 누를 때 */
.stButton > button[kind="primary"]:active {
    transform: translateY(5px);
    box-shadow:
        0 1px 0 rgba(25, 75, 150, 0.65),
        0 2px 5px rgba(0, 0, 0, 0.10) !important;
}

/* 체크박스와 입력창의 모서리 */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 10px;
}

/* 구분선 */
hr {
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}

/* 움직임 최소화 설정을 사용하는 환경에서는 애니메이션 제거 */
@media (prefers-reduced-motion: reduce) {
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 제목
# =========================================================

st.title("📊 YouTube Insight")
st.caption(
    "유튜브 영상 정보 · 댓글 감정 분석 · 채널 인기 영상 · 핵심 요약"
)


# =========================================================
# YouTube API
# =========================================================

def get_api_key():
    try:
        key = st.secrets.get("YOUTUBE_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("YOUTUBE_API_KEY", "")

    return key


@st.cache_resource
def get_youtube_service(api_key):
    return build(
        "youtube",
        "v3",
        developerKey=api_key,
        cache_discovery=False
    )


def get_video_id(url):
    url = url.strip()

    if re.fullmatch(r"[\w-]{11}", url):
        return url

    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower().replace("www.", "")

        if host == "youtu.be":
            video_id = parsed.path.strip("/").split("/")[0]

        elif host in ("youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                video_id = parse_qs(
                    parsed.query
                ).get("v", [""])[0]

            elif parsed.path.startswith(
                ("/shorts/", "/embed/", "/live/")
            ):
                parts = parsed.path.strip("/").split("/")
                video_id = parts[1] if len(parts) > 1 else ""

            else:
                video_id = ""
        else:
            video_id = ""

        if re.fullmatch(r"[\w-]{11}", video_id):
            return video_id

        return None

    except Exception:
        return None


def get_video_info(youtube, video_id):
    response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    ).execute()

    items = response.get("items", [])

    if not items:
        return None

    video = items[0]
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})
    details = video.get("contentDetails", {})

    return {
        "video_id": video_id,
        "title": snippet.get("title", "제목 없음"),
        "channel": snippet.get("channelTitle", "채널 정보 없음"),
        "channel_id": snippet.get("channelId", ""),
        "published_at": snippet.get("publishedAt", "")[:10],
        "description": snippet.get("description", ""),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comment_count": int(stats.get("commentCount", 0)),
        "duration": details.get("duration", ""),
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }


def get_comments(youtube, video_id, limit=300):
    comments = []
    next_page_token = None

    while len(comments) < limit:
        request_count = min(100, limit - len(comments))

        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=request_count,
            order="relevance",
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "작성자": snippet.get("authorDisplayName", ""),
                "댓글": snippet.get("textDisplay", ""),
                "좋아요": snippet.get("likeCount", 0),
                "작성일": snippet.get("publishedAt", "")[:10]
            })

        next_page_token = response.get("nextPageToken")

        if not next_page_token:
            break

    return comments


def get_channel_top_videos(youtube, channel_id, limit=5):
    response = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        type="video",
        order="date",
        maxResults=50
    ).execute()

    video_ids = [
        item["id"]["videoId"]
        for item in response.get("items", [])
        if item.get("id", {}).get("videoId")
    ]

    if not video_ids:
        return pd.DataFrame()

    stats_response = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    ).execute()

    rows = []

    for video in stats_response.get("items", []):
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        current_video_id = video.get("id", "")

        rows.append({
            "영상 제목": snippet.get("title", ""),
            "업로드 날짜": snippet.get("publishedAt", "")[:10],
            "조회수": int(stats.get("viewCount", 0)),
            "좋아요": int(stats.get("likeCount", 0)),
            "영상 링크": (
                f"https://www.youtube.com/watch?v={current_video_id}"
            )
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return (
        df.sort_values("조회수", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


# =========================================================
# 댓글 감정 분석
# =========================================================

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="searle-j/kote_for_easygoing_people",
        top_k=None
    )


def analyze_one_emotion(model, text):
    if not isinstance(text, str) or not text.strip():
        return "분석할 내용 없음", 0.0

    try:
        result = model(text[:500], truncation=True)

        if result and isinstance(result[0], list):
            result = result[0]

        if not result:
            return "분석 불가", 0.0

        best = max(
            result,
            key=lambda item: float(item.get("score", 0))
        )

        return (
            str(best.get("label", "분석 불가")),
            round(float(best.get("score", 0)), 4)
        )

    except Exception:
        return "분석 불가", 0.0


def analyze_comments(comments_df):
    model = load_emotion_model()
    labels = []
    scores = []

    progress = st.progress(0, text="댓글 감정 분석 준비 중...")
    total = len(comments_df)

    for index, comment in enumerate(comments_df["댓글"]):
        label, score = analyze_one_emotion(model, comment)

        labels.append(label)
        scores.append(score)

        progress.progress(
            (index + 1) / total,
            text=f"댓글 감정 분석 중... {index + 1}/{total}"
        )

    progress.empty()

    result = comments_df.copy()
    result["감정"] = labels
    result["감정 점수"] = scores

    return result


# =========================================================
# 영상 핵심 요약
# =========================================================

@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_path):
    model = load_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="ko",
        vad_filter=True
    )

    return " ".join(
        segment.text.strip()
        for segment in segments
        if segment.text.strip()
    )


@st.cache_resource
def load_summary_model():
    return pipeline(
        "text2text-generation",
        model="eenzeenee/t5-small-korean-summarization"
    )


def summarize_chunk(model, text):
    result = model(
        text,
        max_new_tokens=160,
        do_sample=False,
        truncation=True
    )

    return result[0]["generated_text"].strip()


def choose_summary_emoji(text):
    if any(word in text for word in ["주의", "위험", "문제", "실수"]):
        return "⚠️"

    if any(word in text for word in ["방법", "설명", "배우", "공부", "정보"]):
        return "📚"

    if any(word in text for word in ["웃", "재미", "웃기", "즐거"]):
        return "😄"

    if any(word in text for word in ["감동", "감사", "행복", "응원"]):
        return "💛"

    if any(word in text for word in ["음식", "요리", "먹", "빵", "디저트"]):
        return "🍞"

    return "🎬"


def summarize_text(text):
    if not text or not text.strip():
        return "요약할 음성 내용이 없습니다."

    model = load_summary_model()

    chunks = [
        text[index:index + 600]
        for index in range(0, len(text), 600)
    ]

    partial_summaries = []

    for chunk in chunks:
        partial_summaries.append(
            summarize_chunk(model, chunk)
        )

    combined = " ".join(
        item for item in partial_summaries if item
    )

    if len(combined) > 700:
        combined = summarize_chunk(model, combined[:3000])

    sentences = re.split(
        r"(?<=[.!?。！？])\s*",
        combined.strip()
    )
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return combined

    emoji = choose_summary_emoji(combined)

    return "\n".join(
        f"{emoji} {sentence}"
        for sentence in sentences
    )


# =========================================================
# CSV / Excel 파일 생성
# =========================================================

def make_excel_bytes(dataframe, sheet_name="분석 결과"):
    """Excel 파일을 BytesIO로 생성해 다운로드 오류를 방지한다."""
    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name
        )

    return excel_buffer.getvalue()


def make_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8-sig")


# =========================================================
# 사용자 화면
# =========================================================

st.subheader("1. 분석할 유튜브 영상")

video_url = st.text_input(
    "유튜브 영상 링크를 입력하세요",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.subheader("2. 실행할 분석 기능을 선택하세요")

feature_video_info = st.checkbox(
    "영상 정보 분석",
    value=True
)

feature_comments = st.checkbox(
    "댓글 수집 및 감정 분석",
    value=True
)

feature_channel_top = st.checkbox(
    "채널 인기 영상 TOP 5",
    value=True
)

feature_summary = st.checkbox(
    "영상 핵심 요약",
    value=False
)

audio_file = None

if feature_summary:
    audio_file = st.file_uploader(
        "영상에서 추출한 오디오 파일 업로드",
        type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg"]
    )

    st.markdown(
        "[유튜브 링크로 MP3 만드는 링크](https://media.ytmp3.gg/ko/tool/)"
    )

st.divider()

run_analysis = st.button(
    "분석 시작",
    type="primary",
    use_container_width=True
)


# =========================================================
# 분석 실행
# =========================================================

if run_analysis:
    video_id = get_video_id(video_url) if video_url else None

    if not video_id:
        st.error("올바른 유튜브 영상 링크를 입력해 주세요.")

    elif not any([
        feature_video_info,
        feature_comments,
        feature_channel_top,
        feature_summary
    ]):
        st.warning("먼저 분석할 기능을 하나 이상 선택해 주세요.")

    elif feature_summary and audio_file is None:
        st.warning(
            "핵심 요약을 선택했다면 오디오 파일도 업로드해 주세요. "
            "다른 기능만 사용하려면 핵심 요약 체크를 해제하세요."
        )

    else:
        api_key = get_api_key()

        if not api_key and any([
            feature_video_info,
            feature_comments,
            feature_channel_top
        ]):
            st.error(
                "YouTube API 키를 찾을 수 없습니다. "
                "Streamlit Cloud의 앱 설정에서 "
                "YOUTUBE_API_KEY Secrets를 확인해 주세요."
            )
            st.stop()

        youtube = None

        if api_key:
            try:
                youtube = get_youtube_service(api_key)
            except Exception as error:
                st.error(f"YouTube API 연결 실패: {error}")
                st.stop()

        video_info = None

        if youtube and any([
            feature_video_info,
            feature_comments,
            feature_channel_top
        ]):
            try:
                video_info = get_video_info(youtube, video_id)

                if not video_info:
                    st.error(
                        "영상을 찾을 수 없습니다. "
                        "영상 주소나 공개 상태를 확인해 주세요."
                    )
                    st.stop()

            except Exception as error:
                st.error(
                    "영상 정보를 가져오지 못했습니다. "
                    "API 할당량, API 키, 영상 공개 상태를 확인해 주세요."
                )
                st.exception(error)
                st.stop()

        # -------------------------------------------------
        # 영상 정보 분석
        # -------------------------------------------------

        if feature_video_info and video_info:
            st.divider()
            st.header("🎥 영상 정보 분석")
            st.subheader(video_info["title"])

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("조회수", f'{video_info["views"]:,}')
            col2.metric("좋아요", f'{video_info["likes"]:,}')
            col3.metric("댓글 수", f'{video_info["comment_count"]:,}')
            col4.metric(
                "업로드 날짜",
                video_info["published_at"] or "정보 없음"
            )

            st.write(f'**채널:** {video_info["channel"]}')
            st.write(f'**영상 ID:** {video_info["video_id"]}')
            st.markdown(f'[영상 바로 보기]({video_info["url"]})')

            with st.expander("영상 설명 보기"):
                if video_info["description"]:
                    st.write(video_info["description"])
                else:
                    st.write("등록된 영상 설명이 없습니다.")

            video_info_df = pd.DataFrame([{
                "영상 제목": video_info["title"],
                "채널": video_info["channel"],
                "업로드 날짜": video_info["published_at"],
                "조회수": video_info["views"],
                "좋아요": video_info["likes"],
                "댓글 수": video_info["comment_count"],
                "영상 링크": video_info["url"]
            }])

            st.download_button(
                "영상 정보 CSV 다운로드",
                data=make_csv_bytes(video_info_df),
                file_name="youtube_video_info.csv",
                mime="text/csv"
            )

        # -------------------------------------------------
        # 댓글 수집 및 감정 분석
        # -------------------------------------------------

        if feature_comments and youtube:
            st.divider()
            st.header("💬 댓글 수집 및 감정 분석")

            try:
                with st.spinner("댓글을 수집하고 있습니다..."):
                    comments = get_comments(
                        youtube,
                        video_id,
                        limit=300
                    )

                if not comments:
                    st.warning(
                        "수집 가능한 댓글이 없습니다. "
                        "댓글 사용이 제한된 영상일 수 있습니다."
                    )

                else:
                    comments_df = pd.DataFrame(comments)

                    st.write(
                        f"수집한 댓글: **{len(comments_df):,}개**"
                    )

                    with st.spinner(
                        "감정 분석 모델을 불러오고 댓글을 분석합니다..."
                    ):
                        analyzed_df = analyze_comments(comments_df)

                    st.success("댓글 감정 분석이 완료되었습니다.")
                    st.subheader("댓글 분석 결과")

                    st.dataframe(
                        analyzed_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("감정 분포")

                    emotion_counts = (
                        analyzed_df["감정"]
                        .value_counts()
                        .rename_axis("감정")
                        .reset_index(name="댓글 수")
                    )

                    st.bar_chart(
                        emotion_counts.set_index("감정")
                    )

                    st.dataframe(
                        emotion_counts,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "댓글 분석 결과 CSV 다운로드",
                        data=make_csv_bytes(analyzed_df),
                        file_name="youtube_comments_analysis.csv",
                        mime="text/csv"
                    )

                    # 수정된 Excel 다운로드 기능
                    excel_data = make_excel_bytes(
                        analyzed_df,
                        sheet_name="댓글 분석"
                    )

                    st.download_button(
                        "댓글 분석 결과 엑셀 다운로드",
                        data=excel_data,
                        file_name="youtube_comments_analysis.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        )
                    )

            except Exception as error:
                st.error(
                    "댓글 수집 또는 감정 분석 중 오류가 발생했습니다. "
                    "댓글 설정, API 할당량 및 앱 로그를 확인해 주세요."
                )
                st.exception(error)

        # -------------------------------------------------
        # 채널 인기 영상 TOP 5
        # -------------------------------------------------

        if feature_channel_top and video_info and youtube:
            st.divider()
            st.header("🏆 채널 인기 영상 TOP 5")

            try:
                with st.spinner(
                    "채널의 영상 정보를 불러오고 있습니다..."
                ):
                    top_df = get_channel_top_videos(
                        youtube,
                        video_info["channel_id"],
                        limit=5
                    )

                if top_df.empty:
                    st.warning("채널의 영상 목록을 찾지 못했습니다.")

                else:
                    st.dataframe(
                        top_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("조회수 비교")

                    st.bar_chart(
                        top_df.set_index("영상 제목")["조회수"]
                    )

                    st.download_button(
                        "인기 영상 TOP 5 CSV 다운로드",
                        data=make_csv_bytes(top_df),
                        file_name="channel_top5.csv",
                        mime="text/csv"
                    )

                    top_excel_data = make_excel_bytes(
                        top_df,
                        sheet_name="인기 영상 TOP 5"
                    )

                    st.download_button(
                        "인기 영상 TOP 5 엑셀 다운로드",
                        data=top_excel_data,
                        file_name="channel_top5.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        )
                    )

            except Exception as error:
                st.error("채널 인기 영상을 가져오지 못했습니다.")
                st.exception(error)

        # -------------------------------------------------
        # 영상 핵심 요약
        # -------------------------------------------------

        if feature_summary and audio_file is not None:
            st.divider()
            st.header("📝 영상 핵심 요약")

            temp_audio_path = None

            try:
                extension = os.path.splitext(audio_file.name)[1]

                if not extension:
                    extension = ".audio"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp_audio:
                    temp_audio.write(audio_file.getvalue())
                    temp_audio_path = temp_audio.name

                with st.spinner(
                    "음성을 텍스트로 변환하고 있습니다. "
                    "파일 길이에 따라 시간이 걸릴 수 있습니다..."
                ):
                    transcript = transcribe_audio(temp_audio_path)

                if not transcript.strip():
                    st.warning(
                        "음성을 인식하지 못했습니다. "
                        "음성이 포함된 파일인지 확인해 주세요."
                    )

                else:
                    with st.spinner(
                        "영상의 핵심 내용을 요약하고 있습니다..."
                    ):
                        summary = summarize_text(transcript)

                    st.subheader("핵심 요약")
                    st.markdown(summary)

                    st.download_button(
                        "핵심 요약 TXT 다운로드",
                        data=summary.encode("utf-8-sig"),
                        file_name="youtube_video_summary.txt",
                        mime="text/plain"
                    )

            except Exception as error:
                st.error(
                    "음성 변환 또는 요약 중 오류가 발생했습니다. "
                    "업로드 파일 형식과 Streamlit Cloud 로그를 확인해 주세요."
                )
                st.exception(error)

            finally:
                if (
                    temp_audio_path
                    and os.path.exists(temp_audio_path)
                ):
                    os.remove(temp_audio_path)

        st.divider()
        st.success("선택한 분석 작업을 실행했습니다.")
