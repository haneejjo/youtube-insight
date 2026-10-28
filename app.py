
import re
import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

st.set_page_config(
    page_title="YouTube Insight",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Insight")
st.subheader("유튜브 영상 및 댓글 분석")
st.write("유튜브 영상 링크를 입력하고 분석할 정보를 확인하세요.")

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

            # 1. 영상 정보 분석
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

            # 2. 댓글 수집 및 CSV 다운로드
            if comment_analysis or wordcloud_analysis:
                st.markdown("## 댓글 수집")

                try:
                    comment_rows = []
                    next_page_token = None

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

                        csv_data = comments_df.to_csv(
                            index=False,
                            encoding="utf-8-sig"
                        )

                        st.download_button(
                            "댓글 CSV 다운로드",
                            data=csv_data,
                            file_name="youtube_comments.csv",
                            mime="text/csv"
                        )

                    else:
                        st.warning(
                            "수집할 댓글이 없거나 댓글이 비활성화된 영상이에요."
                        )

                except HttpError:
                    st.error(
                        "댓글을 가져오지 못했어요. "
                        "댓글이 비활성화된 영상인지, "
                        "또는 API 요청 제한에 걸렸는지 확인해 주세요."
                    )

            # 3. 채널 인기 영상 분석
            if popular_views or popular_likes:
                st.info(
                    "채널 인기 영상 분석은 다음 단계에서 연결할게요."
                )

        except KeyError:
            st.error(
                "YouTube API 키 설정이 필요해요. "
                "Streamlit 앱 설정의 Secrets에 키를 등록해 주세요."
            )

        except HttpError:
            st.error(
                "YouTube API 요청에 실패했어요. "
                "API 키와 사용량 제한을 확인해 주세요."
            )

        except Exception as e:
            st.error(f"오류가 발생했어요: {e}")
