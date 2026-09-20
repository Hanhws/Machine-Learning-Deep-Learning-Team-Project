"""국가교통정보센터(ITS) CCTV Open API 클라이언트.

GET https://openapi.its.go.kr:9443/cctvInfo
    ?apiKey&type=ex|its&cctvType=1..5&minX&maxX&minY&maxY&getType=json|xml

응답은 getType 과 무관하게 XML 로 오는 경우가 있어 둘 다 파싱한다.
cctvurl 은 24시간만 유효하므로 워커가 만료 전에 다시 조회한다.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class ItsCctv:
    name: str
    lon: float
    lat: float
    url: str
    format: str | None = None
    cctvtype: str | None = None
    resolution: str | None = None
    roadsectionid: str | None = None
    filecreatetime: str | None = None
    road_type: str = "ex"

    def to_dict(self) -> dict:
        return asdict(self)


class ItsError(RuntimeError):
    pass


def _parse(text: str, road_type: str) -> list[ItsCctv]:
    text = text.strip()
    items: list[dict] = []
    if text.startswith("{"):
        payload = json.loads(text)
        resp = payload.get("response", payload)
        if str(resp.get("resultCode", "0")) not in ("0", "SUCCESS", ""):
            raise ItsError(f"ITS 오류 {resp.get('resultCode')}: {resp.get('resultMsg') or resp.get('resultmsg')}")
        data = resp.get("data") or []
        items = data if isinstance(data, list) else [data]
    elif text.startswith("<"):
        root = ET.fromstring(text)
        code = root.findtext("resultCode") or root.findtext("resultcode")
        if code and code not in ("0",):
            raise ItsError(f"ITS 오류 {code}: {root.findtext('resultMsg') or root.findtext('resultmsg')}")
        for d in root.findall("data"):
            items.append({c.tag: (c.text or "") for c in d})
    else:
        raise ItsError(f"알 수 없는 응답 형식: {text[:120]}")

    out: list[ItsCctv] = []
    for it in items:
        try:
            out.append(
                ItsCctv(
                    name=str(it.get("cctvname", "")).strip(),
                    lon=float(it.get("coordx")),
                    lat=float(it.get("coordy")),
                    url=str(it.get("cctvurl", "")).strip(),
                    format=it.get("cctvformat") or None,
                    cctvtype=str(it.get("cctvtype") or "") or None,
                    resolution=it.get("cctvresolution") or None,
                    roadsectionid=it.get("roadsectionid") or None,
                    filecreatetime=it.get("filecreatetime") or None,
                    road_type=road_type,
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def search(
    road_type: str = "ex",
    cctv_type: str = "1",
    min_x: float = 126.0,
    max_x: float = 130.0,
    min_y: float = 33.0,
    max_y: float = 39.0,
    api_key: str | None = None,
) -> list[ItsCctv]:
    s = get_settings()
    key = api_key or s.its_api_key
    if not key:
        raise ItsError("ITS_API_KEY 가 설정되지 않았습니다 (.env 확인).")
    params = {
        "apiKey": key,
        "type": road_type,
        "cctvType": cctv_type,
        "minX": min_x,
        "maxX": max_x,
        "minY": min_y,
        "maxY": max_y,
        "getType": "json",
    }
    with httpx.Client(timeout=25, verify=s.its_verify_ssl) as client:
        r = client.get(s.its_base_url, params=params)
    if r.status_code != 200:
        raise ItsError(f"ITS HTTP {r.status_code}")
    return _parse(r.text, road_type)


def search_both(cctv_type: str = "1", **bbox) -> list[ItsCctv]:
    """고속도로(ex) + 국도(its) 를 함께 조회."""
    out: list[ItsCctv] = []
    errors = []
    for rt in ("ex", "its"):
        try:
            out.extend(search(rt, cctv_type, **bbox))
        except ItsError as e:
            errors.append(str(e))
    if not out and errors:
        raise ItsError("; ".join(errors))
    return out


def refresh_url(name: str, lon: float, lat: float, road_type: str = "ex", cctv_type: str = "1", pad: float = 0.02) -> str | None:
    """이름·좌표로 다시 조회해 새 URL 을 얻는다 (24h 만료 대응)."""
    try:
        items = search(road_type, cctv_type, lon - pad, lon + pad, lat - pad, lat + pad)
    except ItsError as e:
        log.warning("ITS URL 갱신 실패(%s): %s", name, e)
        return None
    best = None
    best_d = 1e9
    for it in items:
        if it.name == name:
            return it.url
        d = (it.lon - lon) ** 2 + (it.lat - lat) ** 2
        if d < best_d:
            best, best_d = it, d
    if best and best_d < 1e-6:
        return best.url
    return None


def parse_route(cctv_name: str) -> str | None:
    """'[경부선] 신갈JC' → '경부선'."""
    if cctv_name.startswith("[") and "]" in cctv_name:
        return cctv_name[1 : cctv_name.index("]")].strip() or None
    return None
