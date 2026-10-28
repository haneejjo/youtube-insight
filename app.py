
import os
import re
import tempfile
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from transformers import pipeline
from faster_whisper import WhisperModel


# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Insight")
st.write("유튜브 영상의 정보와 댓글, 감정, 핵심 내용을 분석합니다.")

API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")


# ==========================================
# 2. 유튜브 영상 ID 추출
# ==========================================
def get_video_id(url):
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname or ""

        if host in ("youtu.be", "www.youtu.be"):
            return parsed.path.strip("/").split("/")[0]

        if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]

            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
                return parts[1]

    except Exception:
        pass

    return None


# ==========================================
# 3. YouTube API 연결
# ==========================================
@st.cache_resource
def get_youtube_service(api_key):
    return build("youtube", "v3", developerKey=api_key)


# ==========================================
# 4. 영상 정보 가져오기
# ==========================================
def get_video_info(youtube, video_id):
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    items = response.get("items", [])
    if not items:
        return None

    item = items[0]
    snippet = item["snippet"]
    stats = item.get("statistics", {})

    return {
        "영상 제목": snippet.get("title", ""),
        "채널명": snippet.get("channelTitle", ""),
        "업로드 날짜": snippet.get("publishedAt", "")[:10],
        "조회수": int(stats.get("viewCount", 0)),
        "좋아요 수": int(stats.get("likeCount", 0)),
        "댓글 수": int(stats.get("commentCount", 0)),
        "채널 ID": snippet.get("channelId", "")
    }


# ==========================================
# 5. 댓글 수집
# ==========================================
def get_comments(youtube, video_id, max_comments=300):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        count = min(100, max_comments - len(comments))

        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=count,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            ).execute()

        except Exception as e:
            st.warning(
                "댓글을 가져오지 못했습니다. "
                "댓글이 비활성화된 영상일 수도 있습니다. "
                f"상세 내용: {e}"
            )
            break

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "작성자": snippet.get("authorDisplayName", ""),
                "댓글": snippet.get("textDisplay", ""),
                "좋아요 수": snippet.get("likeCount", 0),
                "작성 날짜": snippet.get("publishedAt", "")[:10]
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return pd.DataFrame(comments)


# ==========================================
# 6. 댓글 감정 분석
# ==========================================
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="searle-j/kote_for_easygoing_people",
        tokenizer="searle-j/kote_for_easygoing_people"
    )


def analyze_emotions(comments_df):
    if comments_df.empty:
        return comments_df

    model = load_emotion_model()
    result_df = comments_df.copy()
    texts = result_df["댓글"].fillna("").astype(str).tolist()
    labels = []

    progress = st.progress(0, text="댓글 감정 분석 중...")

    for i, comment in enumerate(texts):
        if not comment.strip():
            labels.append("분석할 내용 없음")
        else:
            try:
                result = model(comment[:1000], truncation=True)[0]
                labels.append(
                    f"{result['label']} ({result['score']:.2f})"
                )
            except Exception:
                labels.append("분석 실패")

        progress.progress(
            (i + 1) / len(texts),
            text=f"댓글 감정 분석 중... {i + 1}/{len(texts)}"
        )

    progress.empty()
    result_df["감정 분석"] = labels

    return result_df


# ==========================================
# 7. 채널 인기 영상 TOP 5
# ==========================================
def get_channel_top_videos(youtube, channel_id):
    channel_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    channel_items = channel_response.get("items", [])
    if not channel_items:
        return pd.DataFrame(), pd.DataFrame()

    playlist_id = channel_items[0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    video_ids = []
    next_page_token = None

    # 채널의 최근 영상 최대 50개 확인
    while len(video_ids) < 50:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=min(50, 50 - len(video_ids)),
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    rows = []

    for start in range(0, len(video_ids), 50):
        batch_ids = video_ids[start:start + 50]

        response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch_ids)
        ).execute()

        for item in response.get("items", []):
            stats = item.get("statistics", {})

            rows.append({
                "영상 제목": item["snippet"].get("title", ""),
                "업로드 날짜": item["snippet"].get("publishedAt", "")[:10],
                "조회수": int(stats.get("viewCount", 0)),
                "좋아요 수": int(stats.get("likeCount", 0)),
                "영상 링크": (
                    "https://www.youtube.com/watch?v=" + item["id"]
                )
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df, df

    top_views = df.sort_values(
        "조회수", ascending=False
    ).head(5)

    top_likes = df.sort_values(
        "좋아요 수", ascending=False
    ).head(5)

    return top_views, top_likes


# ==========================================
# 8. 음성 파일을 텍스트로 변환
# ==========================================
@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_file):
    suffix = os.path.splitext(audio_file.name)[1] or ".mp3"

    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "uploaded_audio" + suffix)

        with open(audio_path, "wb") as f:
            f.write(audio_file.getvalue())

        model = load_whisper_model()

        segments, info = model.transcribe(
            audio_path,
            language="ko",
            vad_filter=True
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments
        )

    return transcript.strip()


# ==========================================
# 9. 핵심 요약 모델
# ==========================================
@st.cache_resource
def load_summary_model():
    return pipeline(
        "text2text-generation",
        model="eenzeenee/t5-small-korean-summarization"
    )


def choose_summary_emoji(text):
    """요약 내용에 어울리는 이모티콘을 선택합니다."""
    lower_text = text.lower()

    topics = [
        (["공부", "학교", "교육", "학습", "시험"], "📚"),
        (["음식", "요리", "식사", "레시피", "맛"], "🍽️"),
        (["여행", "관광", "여행지", "방문"], "✈️"),
        (["인공지능", "기술", "컴퓨터", "ai", "로봇"], "🤖"),
        (["경제", "돈", "가격", "비용", "시장"], "💰"),
        (["운동", "건강", "체력", "수면"], "🌿"),
        (["환경", "기후", "자연", "지구"], "🌎"),
        (["음악", "노래", "가수", "공연"], "🎵"),
        (["영화", "드라마", "배우", "연기"], "🎬"),
        (["역사", "사건", "인물", "문화"], "🗺️")
    ]

    for keywords, emoji in topics:
        if any(word in lower_text for word in keywords):
            return emoji

    return "💡"


def summarize_text(text):
    if not text or not text.strip():
        return "요약할 음성이 인식되지 않았어요."

    model = load_summary_model()

    # 긴 대본을 여러 부분으로 나누어 요약합니다.
    chunks = [
        text[i:i + 700]
        for i in range(0, len(text), 700)
    ]

    summaries = []

    for i, chunk in enumerate(chunks):
        result = model(
            chunk,
            max_new_tokens=128,
            do_sample=False,
            truncation=True
        )

        summary_part = result[0]["generated_text"].strip()

        if summary_part:
            summaries.append(summary_part)

    combined = " ".join(summaries).strip()

    if not combined:
        return "핵심 내용을 요약하지 못했어요. 다른 음성 파일로 다시 시도해 주세요."

    # 여러 조각의 요약이 길면 다시 간결하게 정리합니다.
    if len(combined) > 700:
        result = model(
            combined[:3000],
            max_new_tokens=160,
            do_sample=False,
            truncation=True
        )
        combined = result[0]["generated_text"].strip()

    # 반복되는 공백을 정리합니다.
    combined = re.sub(r"\s+", " ", combined).strip()

    # 읽기 쉽게 문장별 핵심 항목으로 표시합니다.
    sentences = re.split(r"(?<=[.!?。])\s+", combined)
    sentences = [s.strip() for s in sentences if s.strip()]

    emoji = choose_summary_emoji(combined)

    # 대본을 그대로 보여 주는 대신 모델이 만든 요약만 표시합니다.
    formatted_lines = [
        f"• {sentence}" for sentence in sentences
    ]

    return (
        f"{emoji} 핵심 내용 요약\n\n"
        + "\n".join(formatted_lines)
    )


# ==========================================
# 10. 영상 링크 및 분석 체크박스
# ==========================================
video_url = st.text_input(
    "분석할 유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.subheader("분석할 항목을 선택하세요")

col1, col2 = st.columns(2)

with col1:
    select_info = st.checkbox("영상 정보 분석", value=True)
    select_comments = st.checkbox("댓글 수집 및 감정 분석", value=True)

with col2:
    select_channel = st.checkbox("채널 인기 영상 TOP 5", value=True)
    select_summary = st.checkbox("영상 핵심 요약", value=True)

analyze_button = st.button(
    "영상 분석 시작",
    type="primary"
)


# ==========================================
# 11. 분석 실행
# ==========================================
if analyze_button:
    if not API_KEY:
        st.error(
            "YouTube API 키가 설정되지 않았습니다. "
            "Streamlit Secrets에서 YOUTUBE_API_KEY를 확인해 주세요."
        )

    elif not any([
        select_info,
        select_comments,
        select_channel,
        select_summary
    ]):
        st.warning("분석할 항목을 하나 이상 선택해 주세요.")

    else:
        video_id = get_video_id(video_url)

        if not video_id:
            st.error("올바른 유튜브 영상 링크를 입력해 주세요.")

        else:
            try:
                with st.spinner("영상 정보를 확인하는 중..."):
                    info = get_video_info(
                        get_youtube_service(API_KEY),
                        video_id
                    )

                if not info:
                    st.error(
                        "영상을 찾지 못했습니다. "
                        "링크와 영상 공개 상태를 확인해 주세요."
                    )

                else:
                    previous_video_id = st.session_state.get("video_id")

                    # 다른 영상으로 바꾸면 이전 영상의 분석 결과를 지웁니다.
                    if previous_video_id != video_id:
                        for key in [
                            "comments_df",
                            "top_views",
                            "top_likes",
                            "transcript",
                            "summary_text"
                        ]:
                            st.session_state.pop(key, None)

                    st.session_state["video_info"] = info
                    st.session_state["video_id"] = video_id

                    st.session_state["selected_features"] = {
                        "info": select_info,
                        "comments": select_comments,
                        "channel": select_channel,
                        "summary": select_summary
                    }

                    st.success("영상 정보를 불러왔습니다.")

            except Exception as e:
                st.error(f"영상 정보를 가져오지 못했습니다: {e}")


# ==========================================
# 12. 선택한 분석 결과 표시
# ==========================================
info = st.session_state.get("video_info")
selected = st.session_state.get("selected_features", {})

if info and selected:
    youtube = get_youtube_service(API_KEY)

    # --------------------------------------
    # 영상 정보
    # --------------------------------------
    if selected.get("info"):
        st.header("📊 영상 정보")
        st.subheader(info["영상 제목"])

        c1, c2, c3 = st.columns(3)
        c1.metric("조회수", f'{info["조회수"]:,}')
        c2.metric("좋아요 수", f'{info["좋아요 수"]:,}')
        c3.metric("댓글 수", f'{info["댓글 수"]:,}')

        st.write("**채널명:**", info["채널명"])
        st.write("**업로드 날짜:**", info["업로드 날짜"])

    # --------------------------------------
    # 댓글 감정 분석
    # --------------------------------------
    if selected.get("comments"):
        st.header("💬 댓글 수집 및 감정 분석")

        if st.button("댓글 분석 실행"):
            with st.spinner("댓글을 수집하는 중..."):
                comments_df = get_comments(
                    youtube,
                    st.session_state["video_id"]
                )

            if comments_df.empty:
                st.session_state["comments_df"] = comments_df
                st.info("수집된 댓글이 없습니다.")

            else:
                with st.spinner("댓글 감정을 분석하는 중..."):
                    analyzed_df = analyze_emotions(comments_df)

                st.session_state["comments_df"] = analyzed_df
                st.success(
                    f"{len(analyzed_df)}개의 댓글을 분석했습니다."
                )

        comments_df = st.session_state.get("comments_df")

        if comments_df is not None and not comments_df.empty:
            st.dataframe(
                comments_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "댓글 분석 결과 CSV 다운로드",
                data=comments_df.to_csv(
                    index=False
                ).encode("utf-8-sig"),
                file_name="youtube_comments_analysis.csv",
                mime="text/csv"
            )

            excel_buffer = tempfile.SpooledTemporaryFile()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:
                comments_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="댓글 분석"
                )

            excel_buffer.seek(0)

            st.download_button(
                "댓글 분석 결과 엑셀 다운로드",
                data=excel_buffer.getvalue(),
                file_name="youtube_comments_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # --------------------------------------
    # 채널 인기 영상 TOP 5
    # --------------------------------------
    if selected.get("channel"):
        st.header("🏆 채널 인기 영상 TOP 5")

        if st.button("채널 인기 영상 불러오기"):
            try:
                with st.spinner("채널 영상을 확인하는 중..."):
                    top_views, top_likes = get_channel_top_videos(
                        youtube,
                        info["채널 ID"]
                    )

                st.session_state["top_views"] = top_views
                st.session_state["top_likes"] = top_likes

            except Exception as e:
                st.error(f"채널 영상 조회 실패: {e}")

        top_views = st.session_state.get("top_views")
        top_likes = st.session_state.get("top_likes")

        if top_views is not None:
            st.subheader("조회수 TOP 5")
            st.dataframe(
                top_views,
                use_container_width=True,
                hide_index=True
            )

        if top_likes is not None:
            st.subheader("좋아요 수 TOP 5")
            st.dataframe(
                top_likes,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------
    # 핵심 요약
    # --------------------------------------
    if selected.get("summary"):
        st.header("📝 영상 핵심 요약")

        st.write("분석할 영상의 음성 파일을 업로드해 주세요.")

        st.caption(
            "MP3 파일 준비하기: "
            "[유튜브 링크로 MP3 만드는 링크](https://media.ytmp3.gg/ko/tool/)"
        )

        audio_file = st.file_uploader(
            "음성 파일 업로드",
            type=[
                "mp3", "wav", "m4a", "aac",
                "flac", "ogg", "wma", "mp4", "webm"
            ],
            help=(
                "음성이 포함된 파일을 업로드하세요. "
                "긴 파일은 처리 시간이 더 걸릴 수 있습니다."
            )
        )

        if audio_file:
            st.success(f"업로드 완료: {audio_file.name}")

            if st.button("음성 분석 및 핵심 요약"):
                try:
                    with st.spinner(
                        "음성을 텍스트로 변환 중입니다. "
                        "파일 길이에 따라 시간이 걸릴 수 있습니다..."
                    ):
                        transcript = transcribe_audio(audio_file)

                    if not transcript:
                        st.warning(
                            "음성에서 텍스트를 인식하지 못했습니다."
                        )
                    else:
                        with st.spinner(
                            "핵심 내용을 요약하고 있습니다..."
                        ):
                            summary = summarize_text(transcript)

                        st.session_state["summary_text"] = summary

                except Exception as e:
                    st.error(f"음성 분석 중 오류가 발생했습니다: {e}")
                    st.info(
                        "오디오 파일 형식과 requirements.txt의 "
                        "라이브러리 버전을 확인해 주세요."
                    )

        # 대본은 표시하지 않고 요약 결과만 표시합니다.
        summary = st.session_state.get("summary_text")

        if summary:
            st.subheader("✨ 핵심 내용")
            st.markdown(summary)

            st.download_button(
                "핵심 요약 TXT 다운로드",
                data=summary,
                file_name="youtube_summary.txt",
                mime="text/plain"
            )
