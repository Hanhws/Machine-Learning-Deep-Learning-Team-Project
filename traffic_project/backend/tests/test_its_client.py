import pytest

from app.services import its_client

XML = """<?xml version='1.0' encoding='UTF-8'?>
<response>
    <coordtype>1</coordtype>
    <datacount>1</datacount>
    <data>
        <roadsectionid/>
        <cctvtype>1</cctvtype>
        <cctvurl>http://cctvsec.ktict.co.kr/2/abc==</cctvurl>
        <cctvresolution/>
        <coordy>37.42889</coordy>
        <cctvformat>HLS</cctvformat>
        <cctvname>[수도권제1순환선] 성남</cctvname>
        <coordx>127.12361</coordx>
    </data>
</response>"""

JSON = '{"response":{"coordtype":1,"datacount":1,"data":[{"cctvtype":"1","cctvurl":"http://x/y","coordx":"127.1","coordy":"37.5","cctvformat":"HLS","cctvname":"[경부선] 신갈"}]}}'


def test_parse_xml():
    items = its_client._parse(XML, "ex")
    assert len(items) == 1
    it = items[0]
    assert it.name == "[수도권제1순환선] 성남" and it.lon == 127.12361 and it.lat == 37.42889
    assert it.url.endswith("abc==") and it.format == "HLS" and it.road_type == "ex"


def test_parse_json():
    items = its_client._parse(JSON, "its")
    assert len(items) == 1 and items[0].name == "[경부선] 신갈" and items[0].road_type == "its"


def test_parse_error_code():
    with pytest.raises(its_client.ItsError):
        its_client._parse('{"response":{"resultCode":"4005","resultMsg":"key"}}', "ex")


def test_parse_route():
    assert its_client.parse_route("[경부선] 신갈JC") == "경부선"
    assert its_client.parse_route("신갈JC") is None
