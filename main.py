from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

import yfinance as yf
import requests
import os
import html
import json

# =========================
# .env 로드
# =========================

load_dotenv()

# =========================
# FastAPI
# =========================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# API KEY
# =========================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# 기본 설정
# =========================

DEFAULT_NEWS_KEYWORDS = [
    "AI 반도체",
    "미국 금리",
    "환율",
    "비트코인",
]

DEFAULT_ASSETS = [
    {
        "symbol": "NASDAQ",
        "name": "NASDAQ",
        "ticker": "^IXIC",
    },
    {
        "symbol": "S&P500",
        "name": "S&P500",
        "ticker": "^GSPC",
    },
    {
        "symbol": "BTC",
        "name": "비트코인",
        "ticker": "BTC-USD",
    },
    {
        "symbol": "USD/KRW",
        "name": "원달러 환율",
        "ticker": "USDKRW=X",
    },
]

DEFAULT_SECTIONS = {
    "weather": True,
    "assets": True,
    "news": True,
    "summary": True,
    "fortune": True,
}

DEFAULT_WEATHER_LOCATIONS = [
    {
        "name": "서울",
        "lat": 37.5665,
        "lon": 126.9780,
        "isMain": True,
    },
    {
        "name": "창원",
        "lat": 35.2281,
        "lon": 128.6811,
        "isMain": False,
    },
]


# =========================
# Settings 파싱
# =========================

def parse_settings(settings: str | None):
    default_settings = {
        "weatherLocations": DEFAULT_WEATHER_LOCATIONS,
        "assets": DEFAULT_ASSETS,
        "newsKeywords": DEFAULT_NEWS_KEYWORDS,
        "sections": DEFAULT_SECTIONS,
    }

    if not settings:
        return default_settings

    try:
        print("RAW SETTINGS:", settings)

        parsed = json.loads(settings)

        sections = {
            **DEFAULT_SECTIONS,
            **(parsed.get("sections") or {}),
        }

        weather_locations = (
            parsed.get("weatherLocations")
            or parsed.get("locations")
            or parsed.get("weather")
            or DEFAULT_WEATHER_LOCATIONS
        )

        raw_assets = (
            parsed.get("assets")
            or parsed.get("selectedAssets")
            or parsed.get("assetSettings")
            or parsed.get("watchAssets")
            or DEFAULT_ASSETS
        )

        raw_news_keywords = (
            parsed.get("newsKeywords")
            or parsed.get("news_keywords")
            or parsed.get("keywords")
            or parsed.get("newsKeywordSettings")
            or parsed.get("selectedNewsKeywords")
            or parsed.get("interests")
            or parsed.get("newsInterests")
            or DEFAULT_NEWS_KEYWORDS
        )

        user_settings = {
            "weatherLocations": normalize_weather_locations(weather_locations),
            "assets": normalize_assets(raw_assets),
            "newsKeywords": normalize_news_keywords(raw_news_keywords),
            "sections": sections,
        }

        print("PARSED SETTINGS:", user_settings)

        return user_settings

    except Exception as e:
        print("Settings parse error:", e)
        return default_settings


def normalize_weather_locations(locations):
    normalized = []

    if not isinstance(locations, list):
        return DEFAULT_WEATHER_LOCATIONS

    for item in locations[:3]:
        if not isinstance(item, dict):
            continue

        name = (
            item.get("name")
            or item.get("city")
            or item.get("label")
            or "미확인 지역"
        )

        raw_lat = (
            item.get("lat")
            if item.get("lat") is not None
            else item.get("latitude")
        )

        raw_lon = (
            item.get("lon")
            if item.get("lon") is not None
            else item.get("longitude")
        )

        try:
            lat = float(raw_lat)
            lon = float(raw_lon)
        except Exception:
            print(f"Invalid weather location ignored: {item}")
            continue

        normalized.append(
            {
                "name": str(name),
                "lat": lat,
                "lon": lon,
                "isMain": bool(item.get("isMain", False)),
            }
        )

    if not normalized:
        return DEFAULT_WEATHER_LOCATIONS

    if len(normalized) == 1:
        normalized[0]["isMain"] = True
        return normalized

    has_main = any(item["isMain"] for item in normalized)

    if not has_main:
        normalized[0]["isMain"] = True

    main_already_set = False

    for item in normalized:
        if item["isMain"] and not main_already_set:
            main_already_set = True
        elif item["isMain"] and main_already_set:
            item["isMain"] = False

    return normalized


def normalize_assets(raw_assets):
    """
    Flutter에서 넘어오는 자산 설정을 표준 형태로 변환한다.

    지원 형태:
    1) ["BTC", "USD/KRW"]
    2) [{"name": "이더리움", "ticker": "ETH-USD", ...}]
    3) [{"symbol": "ETH", "ticker": "ETH-USD"}]

    결과:
    [
      {"symbol": "ETH", "name": "이더리움", "ticker": "ETH-USD"}
    ]
    """

    if not isinstance(raw_assets, list):
        return DEFAULT_ASSETS

    normalized = []

    for item in raw_assets:
        asset = normalize_single_asset(item)

        if asset is None:
            continue

        # 중복 ticker 제거
        already_exists = any(
            existing["ticker"] == asset["ticker"]
            for existing in normalized
        )

        if not already_exists:
            normalized.append(asset)

    if not normalized:
        return DEFAULT_ASSETS

    return normalized


def normalize_single_asset(item):
    if isinstance(item, str):
        value = item.strip()

        if not value:
            return None

        default_asset = find_default_asset(value)

        if default_asset:
            return default_asset

        # 문자열이 ticker라고 가정
        return {
            "symbol": value.upper(),
            "name": value.upper(),
            "ticker": value,
        }

    if isinstance(item, dict):
        ticker = (
            item.get("ticker")
            or item.get("yahooTicker")
            or item.get("code")
            or item.get("symbol")
        )

        name = (
            item.get("name")
            or item.get("label")
            or item.get("symbol")
            or item.get("ticker")
        )

        symbol = (
            item.get("symbol")
            or item.get("name")
            or item.get("ticker")
        )

        if not ticker:
            # ticker가 없으면 name/symbol로 기본 자산 매칭 시도
            candidate = (
                item.get("name")
                or item.get("symbol")
                or item.get("label")
            )

            if candidate:
                default_asset = find_default_asset(str(candidate))

                if default_asset:
                    return default_asset

            return None

        return {
            "symbol": str(symbol).strip(),
            "name": str(name).strip(),
            "ticker": str(ticker).strip(),
        }

    return None


def find_default_asset(value):
    normalized_value = value.strip().lower()

    aliases = {
        "nasdaq": "NASDAQ",
        "나스닥": "NASDAQ",
        "^ixic": "NASDAQ",
        "s&p500": "S&P500",
        "s&p 500": "S&P500",
        "sp500": "S&P500",
        "^gspc": "S&P500",
        "btc": "BTC",
        "비트코인": "BTC",
        "btc-usd": "BTC",
        "usd/krw": "USD/KRW",
        "원달러": "USD/KRW",
        "환율": "USD/KRW",
        "usdkrw=x": "USD/KRW",
    }

    matched_symbol = aliases.get(normalized_value)

    if not matched_symbol:
        return None

    for asset in DEFAULT_ASSETS:
        if asset["symbol"] == matched_symbol:
            return asset

    return None


def normalize_news_keywords(raw_keywords):
    """
    Flutter가 어떤 필드명/형태로 보내도 최대한 뉴스 키워드로 변환한다.
    """

    if not isinstance(raw_keywords, list):
        return DEFAULT_NEWS_KEYWORDS

    keywords = []

    for item in raw_keywords:
        if isinstance(item, str):
            keyword = item.strip()

        elif isinstance(item, dict):
            keyword = str(
                item.get("keyword")
                or item.get("name")
                or item.get("label")
                or item.get("value")
                or item.get("title")
                or ""
            ).strip()

        else:
            keyword = ""

        if keyword and keyword not in keywords:
            keywords.append(keyword)

    if not keywords:
        return DEFAULT_NEWS_KEYWORDS

    return keywords


# =========================
# 미세먼지 상태 변환
# =========================

def get_pm10_status(pm10):
    if pm10 < 30:
        return "좋음"
    elif pm10 < 80:
        return "보통"
    elif pm10 < 150:
        return "나쁨"
    else:
        return "매우나쁨"


def get_pm25_status(pm25):
    if pm25 < 15:
        return "좋음"
    elif pm25 < 35:
        return "보통"
    elif pm25 < 75:
        return "나쁨"
    else:
        return "매우나쁨"


# =========================
# 미세먼지
# =========================

def get_air_quality(lat, lon):
    try:
        if not OPENWEATHER_API_KEY:
            raise Exception("OPENWEATHER_API_KEY가 없습니다.")

        url = (
            "https://api.openweathermap.org/data/2.5/air_pollution"
            f"?lat={lat}"
            f"&lon={lon}"
            f"&appid={OPENWEATHER_API_KEY}"
        )

        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code != 200:
            raise Exception(data)

        components = data["list"][0]["components"]

        pm10 = components["pm10"]
        pm25 = components["pm2_5"]

        return {
            "pm10": round(pm10, 1),
            "pm25": round(pm25, 1),
            "pm10Status": get_pm10_status(pm10),
            "pm25Status": get_pm25_status(pm25),
        }

    except Exception as e:
        print("Air quality error:", e)

        return {
            "pm10": None,
            "pm25": None,
            "pm10Status": "확인 불가",
            "pm25Status": "확인 불가",
        }


# =========================
# 날씨
# =========================

def get_weather_data(weather_locations=None):
    locations = normalize_weather_locations(
        weather_locations or DEFAULT_WEATHER_LOCATIONS
    )

    weather_items = []

    for city in locations:
        try:
            if not OPENWEATHER_API_KEY:
                raise Exception("OPENWEATHER_API_KEY가 없습니다.")

            url = (
                "https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={city['lat']}"
                f"&lon={city['lon']}"
                f"&appid={OPENWEATHER_API_KEY}"
                "&units=metric"
                "&lang=kr"
            )

            response = requests.get(url, timeout=10)
            data = response.json()

            if response.status_code != 200:
                raise Exception(data)

            city_timezone_seconds = data["city"]["timezone"]
            city_timezone = timezone(timedelta(seconds=city_timezone_seconds))

            today = datetime.now(city_timezone).date()
            forecasts_by_date = {}

            for item in data["list"]:
                forecast_time = datetime.fromtimestamp(
                    item["dt"],
                    tz=city_timezone,
                )
                forecast_date = forecast_time.date()

                if forecast_date not in forecasts_by_date:
                    forecasts_by_date[forecast_date] = []

                forecasts_by_date[forecast_date].append(item)

            if today in forecasts_by_date:
                selected_date = today
            else:
                selected_date = sorted(forecasts_by_date.keys())[0]

            selected_forecasts = forecasts_by_date[selected_date]

            temps = [
                item["main"]["temp"]
                for item in selected_forecasts
            ]

            conditions = [
                item["weather"][0]["description"]
                for item in selected_forecasts
            ]

            most_common_condition = Counter(conditions).most_common(1)[0][0]

            weather_items.append(
                {
                    "name": city["name"],
                    "low": round(min(temps)),
                    "high": round(max(temps)),
                    "condition": most_common_condition,
                    "isMain": city.get("isMain", False),
                    "airQuality": get_air_quality(
                        city["lat"],
                        city["lon"],
                    ),
                }
            )

        except Exception as e:
            print(f"Weather error - {city['name']}:", e)

            weather_items.append(
                {
                    "name": city["name"],
                    "low": None,
                    "high": None,
                    "condition": "예보 조회 실패",
                    "isMain": city.get("isMain", False),
                    "airQuality": get_air_quality(
                        city["lat"],
                        city["lon"],
                    ),
                }
            )

    return weather_items


# =========================
# 자산
# =========================

def get_market_data(selected_assets=None):
    assets_to_fetch = normalize_assets(
        selected_assets or DEFAULT_ASSETS
    )

    items = []

    for asset in assets_to_fetch:
        ticker_value = asset["ticker"]

        try:
            ticker = yf.Ticker(ticker_value)
            history = ticker.history(period="2d")

            if len(history) < 2:
                print(f"Not enough market data: {ticker_value}")
                continue

            prev_close = history["Close"].iloc[-2]
            current = history["Close"].iloc[-1]

            change_percent = round(
                ((current - prev_close) / prev_close) * 100,
                2,
            )

            items.append(
                {
                    "symbol": asset["symbol"],
                    "name": asset["name"],
                    "ticker": ticker_value,
                    "changePercent": change_percent,
                }
            )

        except Exception as e:
            print(f"Market error - {ticker_value}:", e)

    return items


# =========================
# 뉴스
# =========================

def clean_html_text(text):
    cleaned = html.unescape(text or "")
    cleaned = cleaned.replace("<b>", "")
    cleaned = cleaned.replace("</b>", "")
    cleaned = cleaned.replace("&quot;", '"')
    cleaned = cleaned.replace("&amp;", "&")
    return cleaned


def fetch_news_by_keyword(keyword, display=2):
    try:
        if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
            raise Exception("NAVER API KEY가 없습니다.")

        url = "https://openapi.naver.com/v1/search/news.json"

        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }

        params = {
            "query": keyword,
            "display": display,
            "sort": "date",
        }

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10,
        )

        data = response.json()

        if response.status_code != 200:
            raise Exception(data)

        return data.get("items", [])

    except Exception as e:
        print(f"News fetch error - {keyword}:", e)
        return []


def get_news_data(keywords=None):
    normalized_keywords = normalize_news_keywords(
        keywords or DEFAULT_NEWS_KEYWORDS
    )

    print("NEWS KEYWORDS USED:", normalized_keywords)

    news_items = []
    used_links = set()

    for keyword in normalized_keywords:
        items = fetch_news_by_keyword(keyword, display=2)

        for item in items:
            link = item.get("originallink") or item.get("link")

            if not link:
                continue

            if link in used_links:
                continue

            used_links.add(link)

            news_items.append(
                {
                    "title": clean_html_text(item.get("title", "")),
                    "summary": clean_html_text(item.get("description", "")),
                    "source": "네이버뉴스",
                    "keyword": keyword,
                    "link": link,
                    "publishedAt": item.get("pubDate", ""),
                }
            )

    return news_items[:6]


# =========================
# GPT 요약
# =========================

def get_main_weather(weather):
    if not weather:
        return None

    for item in weather:
        if item.get("isMain"):
            return item

    return weather[0]


def generate_summary(weather, assets, news):
    try:
        if not OPENAI_API_KEY:
            return "오늘의 브리핑을 준비했습니다."

        main_weather = get_main_weather(weather)

        prompt = f"""
아래 데이터를 기반으로 홈 화면용 Morning Brief를 작성해줘.

[메인 지역 날씨]
{main_weather}

[관심 자산]
{assets}

[뉴스]
{news[:4]}

작성 규칙:
- 한국어
- 최대 2문장
- 최대 120자 내외
- 첫 문장: 날씨 기반 행동 가이드
  예: 우산 챙기기, 마스크 챙기기, 겉옷 챙기기
- 비가 오거나 비 가능성이 있으면 우산 언급
- 미세먼지/초미세먼지가 나쁨 이상이면 마스크 언급
- 최저/최고 기온 차이가 8도 이상이면 가벼운 겉옷 언급
- 둘째 문장: 뉴스/시장 흐름의 임플리케이션
- 기사 제목 나열 금지
- 너무 딱딱한 보고서 말투 금지
"""

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 아침에 짧고 실용적인 브리핑을 제공하는 AI 비서다.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.5,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("GPT summary error:", e)
        return "오늘의 브리핑을 준비했습니다."


# =========================
# Morning Report API
# =========================

@app.get("/morning-report")
def get_morning_report(settings: str | None = None):
    user_settings = parse_settings(settings)
    sections = user_settings["sections"]

    weather = (
        get_weather_data(user_settings["weatherLocations"])
        if sections.get("weather", True)
        else []
    )

    assets = (
        get_market_data(user_settings["assets"])
        if sections.get("assets", True)
        else []
    )

    news = (
        get_news_data(user_settings["newsKeywords"])
        if sections.get("news", True)
        else []
    )

    summary = (
        generate_summary(weather, assets, news)
        if sections.get("summary", True)
        else ""
    )

    return {
        "weather": weather,
        "assets": assets,
        "news": news,
        "summary": summary,
        "generatedAt": datetime.now().isoformat(),
    }