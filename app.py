
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from functools import lru_cache
from transformers import pipeline
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.subheader("유튜브 영상 및 댓글 분석")
st.write("유튜브 영상 링크를 입력하고 분석할 기능을 선택하세요.")


video_url = st.text_input(
    "유튜브 영상 링크",
    placeholder="https://www.youtube.com/watch?v=..."
)


def get_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})"
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
        function_to_apply="sigmoid"
    )


def classify_emotion(label):
    positive_labels = [
        "환영/호의", "감동/감탄", "고마움", "존경",
        "기대감", "뿌듯함", "편안/쾌적", "아껴주는",
        "즐거움/신남", "흐뭇함(귀여움/예쁨)",
        "신기함/관심", "행복", "안심/신뢰", "기쁨"
    ]

    negative_labels = [
        "불평/불만", "지긋지긋", "슬픔", "화남/분노",
        "우쭐댐/무시함", "안타까움/실망", "의심/불신",
        "공포/무서움", "절망", "한심함",
        "역겨움/징그러움", "짜증", "증오/혐오",
        "패배/자기혐오", "귀찮음", "힘듦/지침",
        "죄책감", "재미없음", "부담/안_내킴",
        "서러움", "불쌍함/연민", "불안/걱정"
    ]

    if label in positive_labels:
        return "긍정"
    if label in negative_labels:
        return "부정"
    return "중립"


st.markdown("### 분석할 기능 선택")

video_stats = st.checkbox("영상 정보 분석", value=True)
comment_analysis = st.checkbox("댓글 감정 분석")
wordcloud_analysis = st.checkbox("워드클라우드")
popular_views = st.checkbox("채널 조회수 TOP 5")
popular_likes = st.checkbox("채널 좋아요 TOP 5")


if st.button("분석 시작", type="primary"):
    video_id = get_video_id(video_url.strip())

    if not video_id:
        st.error("올바른 유튜브 영상 링크를 입력해 주세요.")

    elif not any([
        video_stats,
        comment_analysis,
        wordcloud_analysis,
        popular_views,
        popular_likes
    ]):
        st.warning("분석할 기능을 하나 이상 선택해 주세요.")

    else:
        try:
            api_key = st.secrets["YOUTUBE_API_KEY"]

            youtube = build(
                "youtube",
                "v3",
                developerKey=api_key
            )

            # 1. 영상 정보
            if video_stats:
                response = youtube.videos().list(
                    part="snippet,statistics",
                    id=video_id
                ).execute()

                if not response["items"]:
                    st.error("영상을 찾을 수 없어요. 링크를 확인해 주세요.")
                else:
                    item = response["items"][0]
                    snippet = item["snippet"]
                    stats = item["statistics"]

                    st.markdown("## 영상 정보")
                    st.subheader(snippet["title"])
                    st.write("채널:", snippet["channelTitle"])
                    st.write("업로드 날짜:", snippet["publishedAt"][:10])

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "조회수",
                        f'{int(stats.get("viewCount", 0)):,}'
                    )
                    col2.metric(
                        "좋아요",
                        f'{int(stats.get("likeCount", 0)):,}'
                    )
                    col3.metric(
                        "댓글 수",
                        f'{int(stats.get("commentCount", 0)):,}'
                    )

            # 2. 댓글 수집
            comments_df = None

            if comment_analysis or wordcloud_analysis:
                st.markdown("## 댓글 수집")

                comment_rows = []
                next_page_token = None

                try:
                    with st.spinner("댓글을 수집하고 있어요..."):
                        while len(comment_rows) < 300:
                            response = youtube.commentThreads().list(
                                part="snippet",
                                videoId=video_id,
                                maxResults=min(
                                    100,
                                    300 - len(comment_rows)
                                ),
                                pageToken=next_page_token,
                                textFormat="plainText"
                            ).execute()

                            for item in response.get("items", []):
                                comment = item["snippet"][
                                    "topLevelComment"
                                ]["snippet"]

                                comment_rows.append({
                                    "작성자": comment.get(
                                        "authorDisplayName", ""
                                    ),
                                    "댓글": comment.get(
                                        "textDisplay", ""
                                    ),
                                    "좋아요 수": comment.get(
                                        "likeCount", 0
                                    ),
                                    "작성일": comment.get(
                                        "publishedAt", ""
                                    )
                                })

                            next_page_token = response.get(
                                "nextPageToken"
                            )

                            if not next_page_token:
                                break

                    if comment_rows:
                        comments_df = pd.DataFrame(comment_rows)

                        st.success(
                            f"댓글 {len(comments_df)}개를 수집했어요."
                        )
                        st.dataframe(
                            comments_df,
                            use_container_width=True
                        )

                    else:
                        st.warning(
                            "수집할 댓글이 없거나 댓글이 비활성화된 영상이에요."
                        )

                except HttpError:
                    st.error(
                        "댓글 수집에 실패했어요. "
                        "댓글 사용 가능 여부와 API 제한을 확인해 주세요."
                    )

            # 3. 댓글 감정 분석
            if comment_analysis and comments_df is not None:
                st.markdown("## 댓글 감정 분석")

                try:
                    model = load_emotion_model()
                    results = []

                    with st.spinner(
                        "감정을 분석하고 있어요. 처음에는 모델 다운로드로 "
                        "시간이 걸릴 수 있어요."
                    ):
                        texts = comments_df["댓글"].fillna("").tolist()

                        for start in range(0, len(texts), 8):
                            batch = texts[start:start + 8]

                            predictions = model(
                                batch,
                                truncation=True,
                                max_length=256
                            )

                            for text, prediction in zip(
                                batch, predictions
                            ):
                                if not prediction:
                                    label = "없음"
                                    score = 0.0
                                else:
                                    best = max(
                                        prediction,
                                        key=lambda x: x["score"]
                                    )
                                    label = best["label"]
                                    score = best["score"]

                                results.append({
                                    "댓글": text,
                                    "주요 감정": label,
                                    "감정 분류": classify_emotion(label),
                                    "모델 점수": round(float(score), 3)
                                })

                    sentiment_df = pd.DataFrame(results)

                    st.success("댓글 감정 분석이 완료됐어요.")

                    counts = (
                        sentiment_df["감정 분류"]
                        .value_counts()
                        .reindex(
                            ["긍정", "부정", "중립"],
                            fill_value=0
                        )
                    )

                    total = len(sentiment_df)

                    col1, col2, col3 = st.columns(3)

                    for col, label in zip(
                        [col1, col2, col3],
                        ["긍정", "부정", "중립"]
                    ):
                        count = int(counts[label])
                        percent = count / total * 100 if total else 0

                        col.metric(
                            f"{label} 댓글",
                            f"{count}개",
                            f"{percent:.1f}%"
                        )

                    st.markdown("### 감정 비율 그래프")

                    fig, ax = plt.subplots()
                    counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("감정 분류")
                    ax.set_ylabel("댓글 수")
                    ax.set_title("유튜브 댓글 감정 분석")
                    ax.tick_params(axis="x", rotation=0)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown("### 댓글별 분석 결과")
                    st.dataframe(
                        sentiment_df,
                        use_container_width=True
                    )

                    sentiment_csv = sentiment_df.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    )

                    st.download_button(
                        "감정 분석 결과 CSV 다운로드",
                        data=sentiment_csv,
                        file_name="youtube_sentiment_analysis.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(
                        "감정 분석 중 오류가 발생했어요. "
                        "requirements.txt와 배포 로그를 확인해 주세요."
                    )
                    st.code(str(e))

            # 4. 현재 미연결 기능 안내
            if wordcloud_analysis:
                st.info("워드클라우드는 다음 단계에서 연결할게요.")

            if popular_views or popular_likes:
                st.info(
                    "채널 인기 영상 분석은 다음 단계에서 연결할게요."
                )

        except KeyError:
            st.error(
                "YouTube API 키가 필요해요. "
                "Streamlit Secrets 설정을 확인해 주세요."
            )

        except HttpError:
            st.error(
                "YouTube API 요청에 실패했어요. "
                "API 키와 사용량 제한을 확인해 주세요."
            )

        except Exception as e:
            st.error(f"오류가 발생했어요: {e}")
