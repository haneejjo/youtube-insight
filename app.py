
import os
import re
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.write("영상 핵심 요약과 댓글 분석을 한곳에서 확인하세요.")


# -------------------------
# 공통 함수
# -------------------------

def get_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def safe_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def excel_bytes(df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="분석 결과")

    return output.getvalue()


def download_buttons(df, filename):
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "CSV 다운로드",
            data=df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"{filename}.csv",
            mime="text/csv",
            key=f"{filename}_csv"
        )

    with col2:
        st.download_button(
            "엑셀 다운로드",
            data=excel_bytes(df),
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{filename}_xlsx"
        )


# -------------------------
# 업로드한 음성 파일 처리
# -------------------------

@st.cache_resource
def load_whisper_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_path):
    model = load_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        beam_size=3,
        vad_filter=True
    )

    transcript_parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            transcript_parts.append(text)

    transcript_text = " ".join(transcript_parts).strip()

    if not transcript_text:
        raise ValueError(
            "음성을 텍스트로 변환하지 못했습니다. "
            "파일에 음성이 있는지 확인해 주세요."
        )

    return transcript_text, getattr(info, "language", "알 수 없음")


# -------------------------
# 한국어 요약 모델
# -------------------------

@st.cache_resource
def load_summary_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "eenzeenee/t5-small-korean-summarization"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    return tokenizer, model


def summarize_one_chunk(text, tokenizer, model):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()


def summarize_transcript(text, tokenizer, model):
    current_text = text

    for _ in range(4):
        token_ids = tokenizer.encode(
            current_text,
            add_special_tokens=False
        )

        if len(token_ids) <= 400:
            return summarize_one_chunk(
                current_text,
                tokenizer,
                model
            )

        chunks = []

        for start in range(0, len(token_ids), 400):
            chunk_ids = token_ids[start:start + 400]
            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            )

            if chunk_text.strip():
                chunks.append(chunk_text)

        summaries = []

        for chunk in chunks:
            result = summarize_one_chunk(
                chunk,
                tokenizer,
                model
            )

            if result:
                summaries.append(result)

        current_text = " ".join(summaries)

        if not current_text.strip():
            raise ValueError("요약 결과가 비어 있습니다.")

    return summarize_one_chunk(
        current_text,
        tokenizer,
        model
    )


# -------------------------
# 댓글 감정 분석 모델
# -------------------------

@st.cache_resource
def load_emotion_model():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="searle-j/kote_for_easygoing_people",
        top_k=None
    )


def find_predictions(value):
    found = []

    if isinstance(value, dict):
        if "label" in value and "score" in value:
            found.append(value)
        else:
            for item in value.values():
                found.extend(find_predictions(item))

    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(find_predictions(item))

    return found


def classify_sentiment(prediction):
    candidates = find_predictions(prediction)

    if not candidates:
        return "중립", "없음", 0.0

    best = max(
        candidates,
        key=lambda item: float(item.get("score", 0))
    )

    emotion = str(best.get("label", "없음"))
    score = float(best.get("score", 0))

    positive_words = [
        "기쁨", "즐거움", "신남", "감동", "감탄",
        "기대감", "안심", "신뢰", "존경", "사랑",
        "행복", "재미", "칭찬", "감사", "만족"
    ]

    negative_words = [
        "화남", "분노", "불평", "불만", "한심함",
        "짜증", "불안", "걱정", "실망", "슬픔",
        "혐오", "경멸", "두려움", "의심", "불신",
        "안타까움", "비장함"
    ]

    if emotion == "없음":
        sentiment = "중립"
    elif any(word in emotion for word in positive_words):
        sentiment = "긍정"
    elif any(word in emotion for word in negative_words):
        sentiment = "부정"
    else:
        sentiment = "중립"

    return sentiment, emotion, score


# -------------------------
# 사용자 입력
# -------------------------

video_url = st.text_input(
    "유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.subheader("영상 요약용 음성 파일")
st.write(
    "유튜브에서 음성을 자동으로 내려받지 않습니다. "
    "분석할 영상의 음성 파일을 직접 업로드해 주세요."
)

audio_file = st.file_uploader(
    "음성 파일 업로드",
    type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma", "mp4", "webm"],
    help="음성이 포함된 파일을 업로드하세요. 긴 파일은 처리 시간이 더 걸릴 수 있습니다."
)

if audio_file:
    st.success(f"업로드 완료: {audio_file.name}")

st.subheader("분석할 기능")

left, right = st.columns(2)

with left:
    video_stats = st.checkbox("영상 정보 분석", value=True)
    video_summary = st.checkbox("영상 핵심 요약", value=True)
    comment_analysis = st.checkbox("댓글 감정 분석", value=True)

with right:
    popular_views = st.checkbox("채널 조회수 TOP 5", value=True)
    popular_likes = st.checkbox("채널 좋아요 TOP 5", value=True)
    export_results = st.checkbox("CSV·엑셀 다운로드", value=True)

st.caption("댓글은 최대 300개까지 수집합니다.")


# -------------------------
# 분석 시작
# -------------------------

if st.button("전체 분석 시작", type="primary"):

    video_id = get_video_id(video_url.strip())

    if not video_url.strip():
        st.warning("유튜브 영상 링크를 입력해 주세요.")
        st.stop()

    if not video_id:
        st.error("올바른 유튜브 영상 링크를 입력해 주세요.")
        st.stop()

    try:
        api_key = st.secrets["YOUTUBE_API_KEY"]

        youtube = build(
            "youtube",
            "v3",
            developerKey=api_key
        )

    except KeyError:
        st.error(
            "Streamlit Secrets에 YOUTUBE_API_KEY가 "
            "설정되어 있는지 확인해 주세요."
        )
        st.stop()

    except Exception as e:
        st.error(f"YouTube API 연결 실패: {e}")
        st.stop()

    try:
        response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not response.get("items"):
            st.error("영상을 찾을 수 없습니다.")
            st.stop()

        video = response["items"][0]
        snippet = video["snippet"]
        statistics = video["statistics"]
        channel_id = snippet["channelId"]

        # 1. 영상 정보
        if video_stats:
            st.header("1. 영상 정보")
            st.subheader(snippet.get("title", "제목 없음"))
            st.write(f"채널: {snippet.get('channelTitle', '')}")
            st.write(
                f"업로드 날짜: {snippet.get('publishedAt', '')[:10]}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "조회수",
                f"{safe_int(statistics.get('viewCount')):,}"
            )
            c2.metric(
                "좋아요",
                f"{safe_int(statistics.get('likeCount')):,}"
            )
            c3.metric(
                "댓글 수",
                f"{safe_int(statistics.get('commentCount')):,}"
            )

        # 2. 음성 파일 기반 영상 요약
        if video_summary:
            st.header("2. 영상 핵심 요약")

            if audio_file is None:
                st.warning(
                    "영상 요약을 하려면 위에서 음성 파일을 업로드해 주세요. "
                    "음성 파일 없이도 다른 분석 기능은 사용할 수 있습니다."
                )

            else:
                try:
                    # 파일 확장자를 유지해 ffmpeg가 파일 형식을 인식하게 합니다.
                    suffix = os.path.splitext(audio_file.name)[1].lower()

                    if not suffix:
                        suffix = ".audio"

                    with tempfile.TemporaryDirectory() as temp_dir:
                        audio_path = os.path.join(
                            temp_dir,
                            "uploaded_audio" + suffix
                        )

                        with open(audio_path, "wb") as file:
                            file.write(audio_file.getvalue())

                        with st.spinner(
                            "1/2 AI가 음성을 텍스트로 변환하고 있습니다. "
                            "처음에는 모델 다운로드 때문에 오래 걸릴 수 있습니다."
                        ):
                            transcript_text, detected_language = (
                                transcribe_audio(audio_path)
                            )

                        with st.spinner(
                            "2/2 AI가 핵심 내용을 요약하고 있습니다..."
                        ):
                            tokenizer, summary_model = load_summary_model()

                            summary_text = summarize_transcript(
                                transcript_text,
                                tokenizer,
                                summary_model
                            )

                    if summary_text:
                        st.subheader("핵심 요약")
                        st.write(summary_text)

                        st.caption(
                            f"음성 인식 언어: {detected_language}"
                        )

                        st.info(
                            "업로드한 파일의 음성을 바탕으로 요약했습니다. "
                            "화면에만 표시되는 글자나 음성으로 설명되지 않는 "
                            "장면은 포함되지 않을 수 있습니다."
                        )

                        with st.expander("음성 인식 결과 확인"):
                            st.write(transcript_text)

                        if export_results:
                            summary_df = pd.DataFrame([{
                                "영상 제목": snippet.get("title", ""),
                                "영상 링크": video_url.strip(),
                                "음성 인식 언어": detected_language,
                                "음성 인식 내용": transcript_text,
                                "핵심 요약": summary_text
                            }])

                            download_buttons(
                                summary_df,
                                "youtube_video_summary"
                            )

                    else:
                        st.warning("요약 결과가 비어 있습니다.")

                except Exception as e:
                    st.error("음성 파일 분석에 실패했습니다.")
                    st.write(
                        "지원하지 않는 파일이거나 음성 인식 또는 "
                        "요약 모델 처리 중 문제가 발생했을 수 있습니다."
                    )
                    st.code(str(e))

        # 3. 댓글 수집
        comments = []

        if comment_analysis:
            st.header("3. 댓글 수집")

            try:
                with st.spinner("댓글을 수집하고 있습니다..."):
                    next_page_token = None

                    while len(comments) < 300:
                        result = youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=min(100, 300 - len(comments)),
                            pageToken=next_page_token,
                            textFormat="plainText"
                        ).execute()

                        for item in result.get("items", []):
                            comment = (
                                item["snippet"]
                                ["topLevelComment"]
                                ["snippet"].get("textDisplay", "")
                            )

                            if comment.strip():
                                comments.append(comment)

                        next_page_token = result.get("nextPageToken")

                        if not next_page_token:
                            break

                st.success(f"댓글 {len(comments)}개를 수집했습니다.")

            except HttpError as e:
                st.warning(
                    "댓글을 수집하지 못했습니다. "
                    "댓글이 비활성화된 영상일 수 있습니다."
                )
                st.caption(str(e))

        comments_df = pd.DataFrame({"댓글": comments})

        if comments and export_results:
            download_buttons(comments_df, "youtube_comments")

        # 4. 댓글 감정 분석
        if comment_analysis and comments:
            st.header("4. 댓글 감정 분석")

            try:
                model = load_emotion_model()
                emotion_results = []

                progress = st.progress(0)
                status = st.empty()
                batch_size = 8

                with st.spinner("AI가 댓글 감정을 분석하고 있습니다..."):
                    for start in range(0, len(comments), batch_size):
                        batch = comments[start:start + batch_size]

                        predictions = model(
                            batch,
                            truncation=True,
                            max_length=256
                        )

                        for comment, prediction in zip(batch, predictions):
                            sentiment, emotion, score = (
                                classify_sentiment(prediction)
                            )

                            emotion_results.append({
                                "댓글": comment,
                                "감정 분류": sentiment,
                                "세부 감정": emotion,
                                "점수": round(score, 4)
                            })

                        completed = min(start + batch_size, len(comments))
                        progress.progress(completed / len(comments))
                        status.write(
                            f"{completed}/{len(comments)}개 분석 완료"
                        )

                emotion_df = pd.DataFrame(emotion_results)

                st.subheader("댓글별 감정 결과")
                st.dataframe(emotion_df, use_container_width=True)

                st.subheader("긍정·부정·중립 비율")

                counts = (
                    emotion_df["감정 분류"]
                    .value_counts()
                    .reindex(["긍정", "부정", "중립"], fill_value=0)
                )

                st.bar_chart(counts)

                total = max(len(emotion_df), 1)
                a, b, c = st.columns(3)

                a.metric("긍정", f"{counts['긍정'] / total * 100:.1f}%")
                b.metric("부정", f"{counts['부정'] / total * 100:.1f}%")
                c.metric("중립", f"{counts['중립'] / total * 100:.1f}%")

                st.subheader("세부 감정 빈도")
                st.bar_chart(emotion_df["세부 감정"].value_counts())

                if export_results:
                    download_buttons(
                        emotion_df,
                        "youtube_emotion_analysis"
                    )

            except Exception as e:
                st.error("감정 분석 중 오류가 발생했습니다.")
                st.code(str(e))

        # 5. 채널 인기 영상 TOP 5
        if popular_views or popular_likes:
            st.header("5. 채널 인기 영상 TOP 5")

            try:
                channel_response = youtube.search().list(
                    part="snippet",
                    channelId=channel_id,
                    type="video",
                    order="viewCount",
                    maxResults=50
                ).execute()

                channel_videos = channel_response.get("items", [])

                video_ids = [
                    item["id"]["videoId"]
                    for item in channel_videos
                    if item.get("id", {}).get("videoId")
                ]

                video_details = []

                for start in range(0, len(video_ids), 50):
                    detail_response = youtube.videos().list(
                        part="snippet,statistics",
                        id=",".join(video_ids[start:start + 50])
                    ).execute()

                    for item in detail_response.get("items", []):
                        stats = item.get("statistics", {})
                        detail_snippet = item.get("snippet", {})

                        video_details.append({
                            "영상 제목": detail_snippet.get("title", ""),
                            "업로드 날짜": detail_snippet.get(
                                "publishedAt", ""
                            )[:10],
                            "조회수": safe_int(stats.get("viewCount")),
                            "좋아요": safe_int(stats.get("likeCount")),
                            "영상 링크": (
                                "https://www.youtube.com/watch?v="
                                + item["id"]
                            )
                        })

                channel_df = pd.DataFrame(video_details)

                if channel_df.empty:
                    st.info("채널 영상을 찾을 수 없습니다.")
                else:
                    if popular_views:
                        st.subheader("조회수 TOP 5")

                        views_top = channel_df.sort_values(
                            "조회수",
                            ascending=False
                        ).head(5)

                        st.dataframe(views_top, use_container_width=True)

                        if export_results:
                            download_buttons(
                                views_top,
                                "channel_views_top5"
                            )

                    if popular_likes:
                        st.subheader("좋아요 TOP 5")

                        likes_top = channel_df.sort_values(
                            "좋아요",
                            ascending=False
                        ).head(5)

                        st.dataframe(likes_top, use_container_width=True)

                        if export_results:
                            download_buttons(
                                likes_top,
                                "channel_likes_top5"
                            )

                    st.caption(
                        "TOP 5는 YouTube API가 검색한 채널 영상 "
                        "최대 50개를 기준으로 합니다."
                    )

            except HttpError as e:
                st.warning("채널 인기 영상 정보를 가져오지 못했습니다.")
                st.caption(str(e))

    except HttpError as e:
        st.error("YouTube API 요청에 실패했습니다.")
        st.code(str(e))

    except Exception as e:
        st.error("분석 중 오류가 발생했습니다.")
        st.code(str(e))
