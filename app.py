import os
import re
import tempfile
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from faster_whisper import WhisperModel
from transformers import pipeline


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.write("유튜브 영상의 정보, 댓글 감정, 핵심 요약을 분석합니다.")

API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")


# -----------------------------
# 유튜브 영상 ID 추출
# -----------------------------
def get_video_id(url):
    try:
        parsed = urlparse(url.strip())

        if parsed.hostname in ["youtu.be", "www.youtu.be"]:
            return parsed.path.strip("/").split("/")[0]

        if parsed.hostname and (
            parsed.hostname.endswith("youtube.com")
            or parsed.hostname.endswith("youtube-nocookie.com")
        ):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]

            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ["shorts", "embed", "live"]:
                return parts[1]

    except Exception:
        return None

    return None


# -----------------------------
# YouTube API 연결
# -----------------------------
@st.cache_resource
def get_youtube_service(api_key):
    return build("youtube", "v3", developerKey=api_key)


# -----------------------------
# 영상 정보 가져오기
# -----------------------------
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


# -----------------------------
# 댓글 수집: 최대 300개
# -----------------------------
def get_comments(youtube, video_id, max_comments=300):
    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request_count = min(100, max_comments - len(comments))

        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=request_count,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            ).execute()
        except Exception as e:
            st.warning(
                "댓글을 가져오지 못했습니다. 댓글 사용이 중지된 영상일 수 있습니다. "
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


# -----------------------------
# 댓글 감정 분석
# -----------------------------
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
    emotion_labels = []

    progress = st.progress(0, text="댓글 감정 분석 준비 중...")

    for i, comment in enumerate(texts):
        if not comment.strip():
            emotion_labels.append("분석할 내용 없음")
        else:
            try:
                result = model(comment[:1000], truncation=True)[0]
                emotion_labels.append(
                    f"{result['label']} ({result['score']:.2f})"
                )
            except Exception:
                emotion_labels.append("분석 실패")

        progress.progress(
            (i + 1) / len(texts),
            text=f"댓글 감정 분석 중... {i + 1}/{len(texts)}"
        )

    progress.empty()
    result_df["감정 분석"] = emotion_labels
    return result_df


# -----------------------------
# 채널 영상 TOP 5
# -----------------------------
def get_channel_top_videos(youtube, channel_id):
    channel_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    channel_items = channel_response.get("items", [])
    if not channel_items:
        return pd.DataFrame(), pd.DataFrame()

    uploads_playlist = channel_items[0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    video_ids = []
    next_page_token = None

    # 최근 영상 최대 50개를 가져와 조회수/좋아요 순으로 정렬
    while len(video_ids) < 50:
        response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=min(50, 50 - len(video_ids)),
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            video_ids.append(
                item["contentDetails"]["videoId"]
            )

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
                    "https://www.youtube.com/watch?v="
                    + item["id"]
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


# -----------------------------
# 음성 파일 받아쓰기
# -----------------------------
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


# -----------------------------
# 핵심 요약 모델
# -----------------------------
@st.cache_resource
def load_summary_model():
    return pipeline(
        "text2text-generation",
        model="eenzeenee/t5-small-korean-summarization"
    )


def summarize_text(text):
    if not text.strip():
        return "요약할 음성이 인식되지 않았습니다."

    model = load_summary_model()

    # 긴 대본은 여러 부분으로 나눠 요약
    chunks = [
        text[i:i + 700]
        for i in range(0, len(text), 700)
    ]

    summaries = []

    for chunk in chunks:
        try:
            result = model(
                chunk,
                max_new_tokens=128,
                do_sample=False,
                truncation=True
            )
            summaries.append(result[0]["generated_text"].strip())
        except Exception as e:
            st.warning(f"일부 문장 요약에 실패했습니다: {e}")

    combined = " ".join(s for s in summaries if s)

    if not combined:
        return "요약을 만들지 못했습니다."

    # 여러 조각을 요약한 결과가 길면 한 번 더 요약
    if len(combined) > 700:
        try:
            result = model(
                combined[:3000],
                max_new_tokens=160,
                do_sample=False,
                truncation=True
            )
            return result[0]["generated_text"].strip()
        except Exception:
            pass

    return combined


# -----------------------------
# 화면: 영상 링크 입력
# -----------------------------
video_url = st.text_input(
    "분석할 유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)

analyze_button = st.button(
    "영상 분석 시작",
    type="primary"
)

if analyze_button:
    if not API_KEY:
        st.error(
            "YouTube API 키가 설정되어 있지 않습니다. "
            "Streamlit의 Secrets에 YOUTUBE_API_KEY를 설정해 주세요."
        )
    else:
        video_id = get_video_id(video_url)

        if not video_id:
            st.error("올바른 유튜브 영상 링크를 입력해 주세요.")
        else:
            try:
                youtube = get_youtube_service(API_KEY)

                with st.spinner("영상 정보를 가져오는 중..."):
                    info = get_video_info(youtube, video_id)

                if not info:
                    st.error(
                        "영상 정보를 찾지 못했습니다. "
                        "영상 링크와 공개 상태를 확인해 주세요."
                    )
                else:
                    st.session_state["video_info"] = info
                    st.session_state["video_id"] = video_id
                    st.session_state["youtube_api_ready"] = True

            except Exception as e:
                st.error(f"영상 정보를 가져오지 못했습니다: {e}")


# -----------------------------
# 영상 정보 출력
# -----------------------------
if st.session_state.get("video_info"):
    info = st.session_state["video_info"]

    st.header("영상 정보")

    st.subheader(info["영상 제목"])
    col1, col2, col3 = st.columns(3)

    col1.metric("조회수", f'{info["조회수"]:,}')
    col2.metric("좋아요 수", f'{info["좋아요 수"]:,}')
    col3.metric("댓글 수", f'{info["댓글 수"]:,}')

    st.write("**채널명:**", info["채널명"])
    st.write("**업로드 날짜:**", info["업로드 날짜"])

    # 댓글 분석
    st.header("댓글 분석")

    if st.button("댓글 수집 및 감정 분석"):
        with st.spinner("댓글을 수집하는 중..."):
            comments_df = get_comments(
                get_youtube_service(API_KEY),
                st.session_state["video_id"]
            )

        if comments_df.empty:
            st.info("수집된 댓글이 없습니다.")
        else:
            with st.spinner("댓글 감정을 분석하는 중..."):
                analyzed_df = analyze_emotions(comments_df)

            st.session_state["comments_df"] = analyzed_df
            st.success(f"{len(analyzed_df)}개의 댓글을 분석했습니다.")

    if st.session_state.get("comments_df") is not None:
        comments_df = st.session_state["comments_df"]
        st.dataframe(comments_df, use_container_width=True)

        csv_data = comments_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "댓글 분석 결과 CSV 다운로드",
            data=csv_data,
            file_name="youtube_comments_analysis.csv",
            mime="text/csv"
        )

        excel_buffer = tempfile.SpooledTemporaryFile()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
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

    # 채널 인기 영상
    st.header("채널 인기 영상 TOP 5")

    if st.button("채널 인기 영상 불러오기"):
        try:
            with st.spinner("채널 영상을 확인하는 중..."):
                top_views, top_likes = get_channel_top_videos(
                    get_youtube_service(API_KEY),
                    info["채널 ID"]
                )

            st.session_state["top_views"] = top_views
            st.session_state["top_likes"] = top_likes

        except Exception as e:
            st.error(f"채널 영상 조회 실패: {e}")

    if st.session_state.get("top_views") is not None:
        st.subheader("조회수 TOP 5")
        st.dataframe(
            st.session_state["top_views"],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("좋아요 TOP 5")
        st.dataframe(
            st.session_state["top_likes"],
            use_container_width=True,
            hide_index=True
        )

    # 음성 업로드 및 핵심 요약
    st.header("영상 핵심 요약")

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
        help="음성이 포함된 파일을 업로드하세요. 긴 파일은 처리 시간이 더 걸릴 수 있습니다."
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
                    st.warning("음성에서 텍스트를 인식하지 못했습니다.")
                else:
                    st.subheader("음성 인식 결과")
                    st.text_area(
                        "변환된 대본",
                        value=transcript,
                        height=250
                    )

                    with st.spinner("영상의 핵심 내용을 요약 중입니다..."):
                        summary = summarize_text(transcript)

                    st.subheader("영상 핵심 요약")
                    st.write(summary)

                    st.download_button(
                        "대본 TXT 다운로드",
                        data=transcript,
                        file_name="youtube_transcript.txt",
                        mime="text/plain"
                    )

                    st.download_button(
                        "핵심 요약 TXT 다운로드",
                        data=summary,
                        file_name="youtube_summary.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(
                    "음성 분석 중 오류가 발생했습니다. "
                    f"오류 내용: {e}"
                )
                st.info(
                    "파일이 정상적으로 재생되는지, 지원되는 형식인지 확인해 주세요."
                )
