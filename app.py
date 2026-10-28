
import re
from collections import Counter
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
st.write("유튜브 영상과 댓글을 분석하는 통합 분석 도구")


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


def excel_bytes(dataframe):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="분석 결과"
        )
    return output.getvalue()


def download_buttons(dataframe, filename):
    csv_data = dataframe.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "CSV 다운로드",
            data=csv_data,
            file_name=f"{filename}.csv",
            mime="text/csv",
            key=f"{filename}_csv"
        )

    with col2:
        try:
            st.download_button(
                "엑셀 다운로드",
                data=excel_bytes(dataframe),
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"{filename}_xlsx"
            )
        except Exception as e:
            st.warning(f"엑셀 파일 생성 실패: {e}")


# -------------------------
# 감정 분석 모델
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
# 입력 및 분석 항목
# -------------------------

video_url = st.text_input(
    "유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.subheader("분석할 기능")

col1, col2 = st.columns(2)

with col1:
    video_stats = st.checkbox("영상 정보 분석", value=True)
    comment_analysis = st.checkbox("댓글 감정 분석", value=True)
    wordcloud_analysis = st.checkbox("워드클라우드", value=True)

with col2:
    popular_views = st.checkbox("채널 조회수 TOP 5", value=True)
    popular_likes = st.checkbox("채널 좋아요 TOP 5", value=True)
    export_results = st.checkbox("분석 결과 CSV·엑셀 다운로드", value=True)

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
            "YouTube API 키가 없습니다. "
            "Streamlit Secrets에 YOUTUBE_API_KEY를 설정해 주세요."
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
        channel_title = snippet.get("channelTitle", "")

        st.header("1. 영상 정보")

        if video_stats:
            st.subheader(snippet.get("title", "제목 없음"))

            st.write(f"채널: {channel_title}")
            st.write(
                f"업로드 날짜: "
                f"{snippet.get('publishedAt', '')[:10]}"
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

        # -------------------------
        # 댓글 수집
        # -------------------------

        comments = []

        if comment_analysis or wordcloud_analysis:
            st.header("2. 댓글 수집")

            try:
                next_page_token = None

                with st.spinner("댓글을 수집하고 있습니다..."):
                    while len(comments) < 300:
                        request = youtube.commentThreads().list(
                            part="snippet",
                            videoId=video_id,
                            maxResults=min(100, 300 - len(comments)),
                            textFormat="plainText",
                            pageToken=next_page_token
                        )

                        result = request.execute()

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

        if comments:
            with st.expander("수집한 댓글 확인"):
                st.dataframe(
                    comments_df,
                    use_container_width=True
                )

            if export_results:
                download_buttons(comments_df, "youtube_comments")

        # -------------------------
        # 댓글 감정 분석
        # -------------------------

        if comment_analysis and comments:
            st.header("3. 댓글 감정 분석")

            try:
                model = load_emotion_model()
                emotion_results = []

                progress = st.progress(0)
                status = st.empty()
                batch_size = 8

                with st.spinner(
                    "AI 모델로 댓글의 감정을 분석하고 있습니다. "
                    "처음에는 시간이 걸릴 수 있습니다."
                ):
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

                        completed = min(
                            start + batch_size,
                            len(comments)
                        )

                        progress.progress(completed / len(comments))
                        status.write(
                            f"{completed}/{len(comments)}개 분석 완료"
                        )

                emotion_df = pd.DataFrame(emotion_results)

                st.subheader("댓글별 분석 결과")
                st.dataframe(
                    emotion_df,
                    use_container_width=True
                )

                st.subheader("긍정·부정·중립 분포")

                sentiment_counts = (
                    emotion_df["감정 분류"]
                    .value_counts()
                    .reindex(
                        ["긍정", "부정", "중립"],
                        fill_value=0
                    )
                )

                st.bar_chart(sentiment_counts)

                total = max(len(emotion_df), 1)
                a, b, c = st.columns(3)

                a.metric(
                    "긍정",
                    f"{sentiment_counts['긍정'] / total * 100:.1f}%"
                )
                b.metric(
                    "부정",
                    f"{sentiment_counts['부정'] / total * 100:.1f}%"
                )
                c.metric(
                    "중립",
                    f"{sentiment_counts['중립'] / total * 100:.1f}%"
                )

                st.subheader("세부 감정 빈도")

                emotion_counts = emotion_df["세부 감정"].value_counts()
                st.bar_chart(emotion_counts)

                if export_results:
                    download_buttons(
                        emotion_df,
                        "youtube_emotion_analysis"
                    )

            except Exception as e:
                st.error("감정 분석 중 오류가 발생했습니다.")
                st.code(str(e))
                st.info(
                    "모델 다운로드, 메모리 또는 라이브러리 문제일 수 있습니다."
                )

        # -------------------------
        # 워드클라우드
        # -------------------------

        if wordcloud_analysis and comments:
            st.header("4. 댓글 워드클라우드")

            try:
                from wordcloud import WordCloud

                text = " ".join(comments)

                # 한글 폰트 경로 탐색
                import os

                font_candidates = [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ]

                font_path = next(
                    (
                        path for path in font_candidates
                        if os.path.exists(path)
                    ),
                    None
                )

                if font_path is None:
                    st.warning(
                        "서버에서 한글 폰트를 찾지 못해 "
                        "워드클라우드 한글 표시가 제한될 수 있습니다."
                    )

                cloud = WordCloud(
                    font_path=font_path,
                    width=1000,
                    height=500,
                    background_color="white",
                    max_words=100
                ).generate(text)

                st.image(
                    cloud.to_array(),
                    caption="댓글에서 자주 등장하는 단어",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"워드클라우드 생성 실패: {e}")

        # -------------------------
        # 채널 인기 영상 TOP 5
        # -------------------------

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

                for i in range(0, len(video_ids), 50):
                    detail_response = youtube.videos().list(
                        part="snippet,statistics",
                        id=",".join(video_ids[i:i + 50])
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

                        st.dataframe(
                            views_top,
                            use_container_width=True
                        )

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

                        st.dataframe(
                            likes_top,
                            use_container_width=True
                        )

                        if export_results:
                            download_buttons(
                                likes_top,
                                "channel_likes_top5"
                            )

            except HttpError as e:
                st.warning(
                    "채널 인기 영상 정보를 가져오지 못했습니다."
                )
                st.caption(str(e))

    except HttpError as e:
        st.error("YouTube API 요청에 실패했습니다.")
        st.code(str(e))

    except Exception as e:
        st.error("분석 중 예상하지 못한 오류가 발생했습니다.")
        st.code(str(e))
