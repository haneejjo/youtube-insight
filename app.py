
import re
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from transformers import pipeline

st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("📊 YouTube Insight")
st.write("유튜브 영상의 정보와 댓글 감정을 분석하는 웹사이트")

video_url = st.text_input(
    "유튜브 영상 링크를 입력하세요",
    placeholder="https://www.youtube.com/watch?v=..."
)

st.subheader("분석할 항목")

video_stats = st.checkbox("영상 정보 분석", value=True)
comment_analysis = st.checkbox("댓글 감정 분석", value=True)
wordcloud_analysis = st.checkbox("워드클라우드")
popular_views = st.checkbox("채널 조회수 TOP 5")
popular_likes = st.checkbox("채널 좋아요 TOP 5")


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


@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="searle-j/kote_for_easygoing_people",
        top_k=None
    )


def find_predictions(value):
    """모델 결과에서 감정 이름과 점수를 안전하게 찾습니다."""
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
        "행복", "재미", "칭찬", "감사"
    ]

    negative_words = [
        "화남", "분노", "불평", "불만", "한심함",
        "짜증", "불안", "걱정", "실망", "슬픔",
        "혐오", "경멸", "두려움", "의심", "불신"
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


if st.button("분석 시작", type="primary"):

    video_id = get_video_id(video_url.strip())

    if not video_url.strip():
        st.warning("유튜브 영상 링크를 입력해 주세요.")

    elif not video_id:
        st.error("올바른 유튜브 영상 링크를 입력해 주세요.")

    else:
        try:
            api_key = st.secrets["YOUTUBE_API_KEY"]

            youtube = build(
                "youtube",
                "v3",
                developerKey=api_key
            )

            # 영상 정보 가져오기
            response = youtube.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()

            if not response.get("items"):
                st.error("영상을 찾을 수 없습니다. 링크를 확인해 주세요.")

            else:
                video = response["items"][0]
                snippet = video["snippet"]
                statistics = video["statistics"]
                channel_id = snippet["channelId"]

                st.header("영상 정보")

                if video_stats:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**영상 제목**")
                        st.write(snippet.get("title", "정보 없음"))

                        st.write("**채널명**")
                        st.write(snippet.get("channelTitle", "정보 없음"))

                        st.write("**업로드 날짜**")
                        st.write(snippet.get("publishedAt", "")[:10])

                    with col2:
                        st.metric(
                            "조회수",
                            f'{int(statistics.get("viewCount", 0)):,}'
                        )
                        st.metric(
                            "좋아요",
                            f'{int(statistics.get("likeCount", 0)):,}'
                        )
                        st.metric(
                            "댓글 수",
                            f'{int(statistics.get("commentCount", 0)):,}'
                        )

                # 댓글 수집
                comments = []

                if comment_analysis or wordcloud_analysis:
                    st.header("댓글 수집")

                    with st.spinner("댓글을 수집하고 있습니다..."):
                        try:
                            next_page_token = None

                            while len(comments) < 300:
                                result = youtube.commentThreads().list(
                                    part="snippet",
                                    videoId=video_id,
                                    maxResults=min(
                                        100, 300 - len(comments)
                                    ),
                                    pageToken=next_page_token,
                                    textFormat="plainText"
                                ).execute()

                                for item in result.get("items", []):
                                    comment = (
                                        item["snippet"]
                                        ["topLevelComment"]
                                        ["snippet"]["textDisplay"]
                                    )
                                    comments.append(comment)

                                next_page_token = result.get(
                                    "nextPageToken"
                                )

                                if not next_page_token:
                                    break

                            st.success(
                                f"댓글 {len(comments)}개를 수집했습니다."
                            )

                        except HttpError as e:
                            st.warning(
                                "댓글을 가져오지 못했습니다. "
                                "댓글 사용이 제한된 영상일 수 있습니다."
                            )
                            st.caption(str(e))

                # 댓글 표 및 CSV
                comments_df = pd.DataFrame({"댓글": comments})

                if len(comments) > 0:
                    with st.expander("수집한 댓글 보기"):
                        st.dataframe(
                            comments_df,
                            use_container_width=True
                        )

                    st.download_button(
                        "댓글 CSV 다운로드",
                        data=comments_df.to_csv(
                            index=False,
                            encoding="utf-8-sig"
                        ),
                        file_name="youtube_comments.csv",
                        mime="text/csv"
                    )

                # 감정 분석
                if comment_analysis and comments:
                    st.header("댓글 감정 분석")

                    if not st.button(
                        "감정 분석 실행",
                        key="run_emotion_analysis"
                    ):
                        st.info(
                            "아래 '감정 분석 실행' 버튼을 눌러 분석하세요."
                        )
                    else:
                        try:
                            model = load_emotion_model()
                            results = []

                            progress = st.progress(0)
                            status = st.empty()

                            batch_size = 8
                            total = len(comments)

                            for start in range(0, total, batch_size):
                                batch = comments[
                                    start:start + batch_size
                                ]

                                predictions = model(
                                    batch,
                                    truncation=True,
                                    max_length=256
                                )

                                for comment, prediction in zip(
                                    batch, predictions
                                ):
                                    sentiment, emotion, score = (
                                        classify_sentiment(prediction)
                                    )

                                    results.append({
                                        "댓글": comment,
                                        "감정 분류": sentiment,
                                        "세부 감정": emotion,
                                        "점수": round(score, 4)
                                    })

                                completed = min(
                                    start + batch_size, total
                                )
                                progress.progress(completed / total)
                                status.write(
                                    f"{completed}/{total}개 댓글 분석 완료"
                                )

                            result_df = pd.DataFrame(results)

                            st.subheader("감정 분석 결과")
                            st.dataframe(
                                result_df,
                                use_container_width=True
                            )

                            st.subheader("긍정·부정·중립 비율")

                            counts = (
                                result_df["감정 분류"]
                                .value_counts()
                                .reindex(
                                    ["긍정", "부정", "중립"],
                                    fill_value=0
                                )
                            )

                            st.bar_chart(counts)

                            col1, col2, col3 = st.columns(3)

                            total_results = len(result_df)

                            with col1:
                                st.metric(
                                    "긍정",
                                    f"{(counts['긍정'] / total_results * 100):.1f}%"
                                )

                            with col2:
                                st.metric(
                                    "부정",
                                    f"{(counts['부정'] / total_results * 100):.1f}%"
                                )

                            with col3:
                                st.metric(
                                    "중립",
                                    f"{(counts['중립'] / total_results * 100):.1f}%"
                                )

                            st.download_button(
                                "감정 분석 결과 CSV 다운로드",
                                data=result_df.to_csv(
                                    index=False,
                                    encoding="utf-8-sig"
                                ),
                                file_name="youtube_emotion_analysis.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error("감정 분석 중 오류가 발생했습니다.")
                            st.code(str(e))
                            st.info(
                                "오류 문구를 보내 주면 원인을 확인할 수 있습니다."
                            )

                if wordcloud_analysis:
                    st.info(
                        "워드클라우드 기능은 다음 단계에서 연결할 수 있습니다."
                    )

                if popular_views or popular_likes:
                    st.info(
                        "채널 인기 영상 TOP 5 기능은 다음 단계에서 연결할 수 있습니다."
                    )

        except KeyError:
            st.error(
                "YouTube API 키를 찾을 수 없습니다. "
                "Streamlit Secrets에 YOUTUBE_API_KEY가 있는지 확인해 주세요."
            )

        except HttpError as e:
            st.error("YouTube API 요청에 실패했습니다.")
            st.code(str(e))

        except Exception as e:
            st.error("오류가 발생했습니다.")
            st.code(str(e))
