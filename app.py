
import os
import tempfile
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from faster_whisper import WhisperModel
from transformers import pipeline


# =====================================
# 기본 설정
# =====================================
st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.write("유튜브 영상의 정보, 댓글, 감정 및 핵심 내용을 분석합니다.")

API_KEY = st.secrets.get("YOUTUBE_API_KEY", "")


# =====================================
# 유튜브 영상 ID 추출
# =====================================
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


# =====================================
# YouTube API
# =====================================
@st.cache_resource
def get_youtube_service(api_key):
    return build("youtube", "v3", developerKey=api_key)


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


# =====================================
# 댓글 수집
# =====================================
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
            st.warning(f"댓글을 가져오지 못했습니다: {e}")
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


# =====================================
# 댓글 감정 분석
# =====================================
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
    labels = []
    progress = st.progress(0, text="댓글 감정 분석 중...")

    for i, comment in enumerate(
        result_df["댓글"].fillna("").astype(str).tolist()
    ):
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
            (i + 1) / len(result_df),
            text=f"댓글 감정 분석 중... {i + 1}/{len(result_df)}"
        )

    progress.empty()
    result_df["감정 분석"] = labels
    return result_df


# =====================================
# 채널 영상 TOP 5
# =====================================
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
        batch = video_ids[start:start + 50]

        response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch)
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


# =====================================
# 음성 파일 받아쓰기
# =====================================
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
        audio_path = os.path.join(temp_dir, "audio" + suffix)

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


# =====================================
# 핵심 요약
# =====================================
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
            st.warning(f"일부 내용의 요약에 실패했습니다: {e}")

    combined = " ".join(s for s in summaries if s)

    if not combined:
        return "요약을 만들지 못했습니다."

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


# =====================================
# 화면 1: 링크 및 분석 항목 선택
# =====================================
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


# =====================================
# 화면 2: 분석 시작
# =====================================
if analyze_button:
    if not API_KEY:
        st.error(
            "YouTube API 키가 설정되지 않았습니다. "
            "Streamlit Secrets의 YOUTUBE_API_KEY를 확인하세요."
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
                    st.error("영상을 찾지 못했습니다. 링크를 확인해 주세요.")

                else:
                    st.session_state["video_info"] = info
                    st.session_state["video_id"] = video_id

                    st.session_state["selected_features"] = {
                        "info": select_info,
                        "comments": select_comments,
                        "channel": select_channel,
                        "summary": select_summary
                    }

                    # 새 영상 분석 시 이전 영상 결과 초기화
                    for key in [
                        "comments_df",
                        "top_views",
                        "top_likes",
                        "transcript",
                        "summary_text"
                    ]:
                        st.session_state.pop(key, None)

                    st.success("영상 정보를 불러왔습니다.")

            except Exception as e:
                st.error(f"영상 정보를 가져오지 못했습니다: {e}")


# =====================================
# 화면 3: 선택된 분석 결과
# =====================================
info = st.session_state.get("video_info")
selected = st.session_state.get("selected_features", {})

if info and selected:

    youtube = get_youtube_service(API_KEY)

    # ---------------------------------
    # 영상 정보
    # ---------------------------------
    if selected.get("info"):
        st.header("영상 정보")
        st.subheader(info["영상 제목"])

        c1, c2, c3 = st.columns(3)
        c1.metric("조회수", f'{info["조회수"]:,}')
        c2.metric("좋아요 수", f'{info["좋아요 수"]:,}')
        c3.metric("댓글 수", f'{info["댓글 수"]:,}')

        st.write("**채널명:**", info["채널명"])
        st.write("**업로드 날짜:**", info["업로드 날짜"])

    # ---------------------------------
    # 댓글 감정 분석
    # ---------------------------------
    if selected.get("comments"):
        st.header("댓글 수집 및 감정 분석")

        if st.button("댓글 분석 실행"):
            with st.spinner("댓글을 수집하는 중..."):
                comments_df = get_comments(
                    youtube,
                    st.session_state["video_id"]
                )

            if comments_df.empty:
                st.info("수집된 댓글이 없습니다.")
                st.session_state["comments_df"] = comments_df
            else:
                with st.spinner("댓글 감정을 분석하는 중..."):
                    analyzed_df = analyze_emotions(comments_df)

                st.session_state["comments_df"] = analyzed_df
                st.success(f"{len(analyzed_df)}개의 댓글을 분석했습니다.")

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

    # ---------------------------------
    # 채널 인기 영상
    # ---------------------------------
    if selected.get("channel"):
        st.header("채널 인기 영상 TOP 5")

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

    # ---------------------------------
    # 영상 핵심 요약
    # ---------------------------------
    if selected.get("summary"):
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
                        st.session_state["transcript"] = transcript

                        with st.spinner("핵심 내용을 요약 중입니다..."):
                            summary = summarize_text(transcript)

                        st.session_state["summary_text"] = summary

                except Exception as e:
                    st.error(f"음성 분석 중 오류가 발생했습니다: {e}")
                    st.info(
                        "음성 파일이 정상적으로 재생되는지, "
                        "requirements.txt의 라이브러리 설정이 맞는지 확인해 주세요."
                    )

        transcript = st.session_state.get("transcript")
        summary = st.session_state.get("summary_text")

        if transcript:
            st.subheader("음성 인식 결과")
            st.text_area(
                "변환된 대본",
                value=transcript,
                height=250
            )

            st.download_button(
                "대본 TXT 다운로드",
                data=transcript,
                file_name="youtube_transcript.txt",
                mime="text/plain"
            )

        if summary:
            st.subheader("영상 핵심 요약")
            st.write(summary)

            st.download_button(
                "핵심 요약 TXT 다운로드",
                data=summary,
                file_name="youtube_summary.txt",
                mime="text/plain"
            )
