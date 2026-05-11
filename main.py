from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI

import yfinance as yf
import requests
import os
import html
import json
import re
from urllib.parse import quote
from email.utils import parsedate_to_datetime

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

MARKET_FALLBACK_CHANGES = {
    "^IXIC": 1.2,
    "^GSPC": 0.8,
    "^DJI": -0.2,
    "KORU": 2.6,
    "BTC-USD": 2.9,
    "ETH-USD": 1.7,
    "USDKRW=X": 0.5,
}

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
        "eth": "ETH",
        "이더리움": "ETH",
        "ethereum": "ETH",
        "eth-usd": "ETH",
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

    if matched_symbol == "ETH":
        return {
            "symbol": "ETH",
            "name": "이더리움",
            "ticker": "ETH-USD",
        }

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

def summarize_daily_weather(forecasts, air_quality):
    """
    OpenWeather 3-hour forecast list for one local date를 하루 단위로 요약한다.
    특정 시간대 description 하나가 아니라 오늘 전체 대비 관점의 대표 상태를 만든다.
    """

    weather_entries = [
        item.get("weather", [{}])[0]
        for item in forecasts
        if item.get("weather")
    ]
    main_values = [
        str(entry.get("main", "")).lower()
        for entry in weather_entries
    ]
    description_values = [
        str(entry.get("description", "")).lower()
        for entry in weather_entries
    ]
    joined = " ".join(main_values + description_values)

    rain_indexes = [
        index
        for index, value in enumerate(main_values + description_values)
        if any(token in value for token in ["rain", "drizzle", "thunderstorm", "비", "소나기", "실비"])
    ]
    snow_indexes = [
        index
        for index, value in enumerate(main_values + description_values)
        if any(token in value for token in ["snow", "눈"])
    ]
    cloudy_count = sum(
        1
        for value in main_values + description_values
        if any(token in value for token in ["cloud", "overcast", "흐림", "구름"])
    )
    clear_count = sum(
        1
        for value in main_values + description_values
        if any(token in value for token in ["clear", "맑음"])
    )

    if rain_indexes:
        condition = "비 가능성"
        display_text = _rain_display_text(forecasts)
    elif snow_indexes:
        condition = "눈 가능성"
        display_text = "오늘 눈 가능성"
    elif cloudy_count > clear_count:
        condition = "흐림"
        display_text = "대체로 흐림"
    elif clear_count >= cloudy_count and clear_count > 0:
        condition = "맑음"
        display_text = "대체로 맑음"
    elif "mist" in joined or "fog" in joined or "안개" in joined:
        condition = "흐림"
        display_text = "안개 또는 흐림"
    else:
        fallback = description_values[0] if description_values else "확인 불가"
        condition = fallback
        display_text = fallback

    pm10_status = air_quality.get("pm10Status")
    pm25_status = air_quality.get("pm25Status")
    if _is_bad_air(pm10_status) or _is_bad_air(pm25_status):
        if condition == "맑음":
            display_text = "맑지만 미세먼지 나쁨"
        else:
            display_text = f"{display_text} · 미세먼지 나쁨"

    return naturalize_weather_text(condition), naturalize_weather_text(display_text)


def _rain_display_text(forecasts):
    rain_times = []
    for item in forecasts:
        weather = item.get("weather", [{}])[0]
        text = f"{weather.get('main', '')} {weather.get('description', '')}".lower()
        if any(token in text for token in ["rain", "drizzle", "thunderstorm", "비", "소나기", "실비"]):
            rain_times.append(item)

    if not rain_times:
        return "비 가능성"

    first = rain_times[0]
    hour_text = ""
    if first.get("dt"):
        try:
            kst = timezone(timedelta(hours=9))
            hour = datetime.fromtimestamp(
                first["dt"],
                tz=timezone.utc,
            ).astimezone(kst).hour
            if 5 <= hour < 12:
                hour_text = "오전 "
            elif 12 <= hour < 18:
                hour_text = "오후 "
            elif 18 <= hour < 24:
                hour_text = "저녁 "
        except Exception:
            hour_text = ""

    rain_amounts = [
        item.get("rain", {}).get("3h", 0) or 0
        for item in rain_times
    ]
    max_rain = max(rain_amounts) if rain_amounts else 0
    strength = "약한 " if max_rain and max_rain < 3 else ""
    return f"{hour_text}{strength}비 가능성".strip()


def _is_bad_air(status):
    return status in ["나쁨", "매우나쁨"]


def naturalize_weather_text(text):
    if not text:
        return "확인 불가"
    return (
        str(text)
        .replace("튼구름", "구름 조금")
        .replace("실 비", "약한 비 가능성")
        .replace("실비", "약한 비 가능성")
        .replace("온흐림", "대체로 흐림")
        .strip()
    )


def weather_icon_key(condition):
    key = str(condition or "").lower().replace(" ", "")
    if any(token in key for token in ["snow", "눈"]):
        return "snow"
    if any(token in key for token in ["rain", "drizzle", "thunderstorm", "비", "소나기"]):
        return "rain"
    if any(token in key for token in ["mist", "fog", "안개"]):
        return "mist"
    if any(token in key for token in ["cloud", "overcast", "흐림", "구름"]):
        return "clouds"
    if any(token in key for token in ["clear", "맑"]):
        return "clear"
    return "clear"


def build_hourly_forecast(forecasts):
    hourly = []
    kst = timezone(timedelta(hours=9))
    for item in forecasts:
        dt = item.get("dt")
        if dt is None:
            continue
        forecast_time = datetime.fromtimestamp(
            dt,
            tz=timezone.utc,
        ).astimezone(kst)
        if forecast_time.hour < 6:
            continue

        weather = item.get("weather", [{}])[0]
        raw_condition = (
            weather.get("description")
            or weather.get("main")
            or "확인 불가"
        )
        condition = naturalize_weather_text(raw_condition)
        temperature = item.get("main", {}).get("temp")
        hourly.append(
            {
                "time": f"{forecast_time.hour:02d}시",
                "temperature": round(temperature) if temperature is not None else None,
                "condition": condition,
                "icon": weather_icon_key(condition),
            }
        )
    return hourly


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

            kst = timezone(timedelta(hours=9))
            today = datetime.now(kst).date()
            forecasts_by_date = {}

            for item in data["list"]:
                forecast_time = datetime.fromtimestamp(
                    item["dt"],
                    tz=timezone.utc,
                ).astimezone(kst)
                forecast_date = forecast_time.date()

                if forecast_date not in forecasts_by_date:
                    forecasts_by_date[forecast_date] = []

                forecasts_by_date[forecast_date].append(item)

            if today in forecasts_by_date:
                selected_date = today
            else:
                selected_date = sorted(forecasts_by_date.keys())[0]

            selected_forecasts = forecasts_by_date[selected_date]

            lows = [
                item.get("main", {}).get("temp_min", item.get("main", {}).get("temp"))
                for item in selected_forecasts
                if item.get("main", {}).get("temp_min", item.get("main", {}).get("temp")) is not None
            ]
            highs = [
                item.get("main", {}).get("temp_max", item.get("main", {}).get("temp"))
                for item in selected_forecasts
                if item.get("main", {}).get("temp_max", item.get("main", {}).get("temp")) is not None
            ]

            if not lows or not highs:
                raise Exception("오늘 예보 기온 데이터가 없습니다.")

            air_quality = get_air_quality(
                city["lat"],
                city["lon"],
            )
            condition, display_text = summarize_daily_weather(
                selected_forecasts,
                air_quality,
            )
            hourly_forecast = build_hourly_forecast(selected_forecasts)

            weather_items.append(
                {
                    "name": city["name"],
                    "low": round(min(lows)),
                    "high": round(max(highs)),
                    "condition": condition,
                    "displayText": display_text,
                    "isMain": city.get("isMain", False),
                    "hourlyForecast": hourly_forecast,
                    "airQuality": air_quality,
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
                    "displayText": "예보 조회 실패",
                    "isMain": city.get("isMain", False),
                    "hourlyForecast": [],
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
            history = ticker.history(period="5d")
            history = history.dropna(subset=["Close"])

            if len(history) < 2:
                print(f"Not enough market data: {ticker_value}")
                items.append(fallback_market_item(asset))
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
                    "price": round(float(current), 2),
                    "changePercent": change_percent,
                    "yahooUrl": yahoo_finance_url(ticker_value),
                }
            )

        except Exception as e:
            print(f"Market error - {ticker_value}:", e)
            items.append(fallback_market_item(asset))

    return items


def fallback_market_item(asset):
    ticker_value = asset["ticker"]
    return {
        "symbol": asset["symbol"],
        "name": asset["name"],
        "ticker": ticker_value,
        "price": None,
        "changePercent": MARKET_FALLBACK_CHANGES.get(ticker_value, 0.0),
        "yahooUrl": yahoo_finance_url(ticker_value),
    }


def yahoo_finance_url(ticker):
    return f"https://finance.yahoo.com/quote/{quote(str(ticker), safe='')}"


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


CORE_MARKET_TERMS = [
    "ai",
    "반도체",
    "엔비디아",
    "nvidia",
    "금리",
    "연준",
    "fomc",
    "환율",
    "달러",
    "원화",
    "나스닥",
    "s&p",
    "비트코인",
    "bitcoin",
    "물가",
    "cpi",
    "채권",
    "유가",
    "실적",
]

LIFE_IMPACT_TERMS = [
    "날씨",
    "미세먼지",
    "교통",
    "파업",
    "요금",
    "물가",
    "유가",
    "전기",
    "가스",
]

CLICKBAIT_TERMS = [
    "충격",
    "경악",
    "대박",
    "난리",
    "헉",
    "무슨 일",
    "알고보니",
    "초비상",
    "단독?",
]

LOW_VALUE_TERMS = [
    "포토",
    "영상",
    "화보",
    "오늘의 운세",
    "로또",
    "행사",
    "이벤트",
    "쿠폰",
]


def _normalize_news_text(value):
    return re.sub(r"\s+", " ", clean_html_text(value)).strip()


def _normalize_for_dedupe(value):
    text = clean_html_text(value).lower()
    text = re.sub(r"\[[^\]]+\]|【[^】]+】|\([^)]*\)", " ", text)
    text = re.sub(r"[^0-9a-z가-힣]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _news_tokens(value):
    normalized = _normalize_for_dedupe(value)
    return {
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in {"뉴스", "기자", "속보", "종합"}
    }


def _published_datetime(value):
    try:
        return parsedate_to_datetime(value or "").astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc) - timedelta(days=365)


def _contains_any(text, terms):
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _keyword_terms(keyword):
    terms = [keyword.strip().lower()]
    terms.extend(
        term
        for term in re.split(r"\s+|/|,", keyword.lower())
        if len(term.strip()) >= 2
    )
    return list(dict.fromkeys(term for term in terms if term))


def _news_relevance_score(item, keywords):
    title = item["title"].lower()
    summary = item["summary"].lower()
    content = f"{title} {summary} {item['keyword'].lower()}"
    score = 0

    for keyword in keywords:
        for term in _keyword_terms(keyword):
            if term in title:
                score += 12
            if term in summary:
                score += 5
            if term == item["keyword"].lower():
                score += 4

    for term in CORE_MARKET_TERMS:
        if term in title:
            score += 7
        elif term in content:
            score += 3

    for term in LIFE_IMPACT_TERMS:
        if term in title:
            score += 3
        elif term in content:
            score += 1

    age_hours = max(
        0,
        (
            datetime.now(timezone.utc)
            - _published_datetime(item.get("publishedAt"))
        ).total_seconds()
        / 3600,
    )
    if age_hours <= 12:
        score += 4
    elif age_hours <= 24:
        score += 2
    elif age_hours > 72:
        score -= 8

    if _contains_any(title, CLICKBAIT_TERMS):
        score -= 8
    if _contains_any(content, LOW_VALUE_TERMS):
        score -= 10
    if len(title) < 12:
        score -= 4
    if not item.get("summary"):
        score -= 3

    return score


def _is_low_value_news(item, keywords):
    title = item["title"]
    summary = item["summary"]
    content = f"{title} {summary}"
    has_keyword_signal = any(
        term in content.lower()
        for keyword in keywords
        for term in _keyword_terms(keyword)
    )
    if not has_keyword_signal and not _contains_any(content, CORE_MARKET_TERMS):
        return True
    if _contains_any(title, LOW_VALUE_TERMS):
        return True
    if _contains_any(title, CLICKBAIT_TERMS) and not _contains_any(
        content,
        CORE_MARKET_TERMS,
    ):
        return True
    return False


def _is_duplicate_news(item, selected):
    title_key = _normalize_for_dedupe(item["title"])
    item_tokens = _news_tokens(item["title"])
    for existing in selected:
        if item.get("link") and item.get("link") == existing.get("link"):
            return True
        existing_key = _normalize_for_dedupe(existing["title"])
        if title_key == existing_key:
            return True
        existing_tokens = _news_tokens(existing["title"])
        if not item_tokens or not existing_tokens:
            continue
        similarity = len(item_tokens & existing_tokens) / len(
            item_tokens | existing_tokens
        )
        if similarity >= 0.62:
            return True
    return False


def _select_briefing_news(candidates, keywords, limit=6, per_keyword_limit=2):
    scored = []
    for item in candidates:
        if _is_low_value_news(item, keywords):
            continue
        score = _news_relevance_score(item, keywords)
        if score <= 0:
            continue
        item["relevanceScore"] = score
        scored.append(item)

    scored.sort(
        key=lambda item: (
            item.get("relevanceScore", 0),
            _published_datetime(item.get("publishedAt")),
        ),
        reverse=True,
    )

    selected = []
    keyword_counts = {}

    for item in scored:
        keyword = item.get("keyword") or ""
        if keyword_counts.get(keyword, 0) >= per_keyword_limit:
            continue
        if _is_duplicate_news(item, selected):
            continue
        selected.append(item)
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        if len(selected) >= limit:
            break

    if len(selected) < min(3, limit):
        for item in scored:
            if _is_duplicate_news(item, selected):
                continue
            selected.append(item)
            if len(selected) >= min(3, limit):
                break

    return selected[:limit]


def get_news_data(keywords=None):
    normalized_keywords = normalize_news_keywords(
        keywords or DEFAULT_NEWS_KEYWORDS
    )

    print("NEWS KEYWORDS USED:", normalized_keywords)

    candidates = []
    used_links = set()

    for keyword in normalized_keywords:
        items = fetch_news_by_keyword(keyword, display=8)

        for item in items:
            link = item.get("originallink") or item.get("link")

            if not link or link in used_links:
                continue

            used_links.add(link)

            candidates.append(
                {
                    "title": _normalize_news_text(item.get("title", "")),
                    "summary": _normalize_news_text(item.get("description", "")),
                    "source": "네이버뉴스",
                    "keyword": keyword,
                    "link": link,
                    "publishedAt": item.get("pubDate", ""),
                }
            )

    return _select_briefing_news(
        candidates,
        normalized_keywords,
        limit=6,
        per_keyword_limit=2,
    )


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
