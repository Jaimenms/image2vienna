"""Eval cases from EUIPO's Trademark Search API.

EUIPO publishes its register through ``api.euipo.europa.eu`` (free developer account
at ``dev.euipo.europa.eu``: create an application, subscribe to the *Trademark
search* product, and put the client id and secret in ``EUIPO_CLIENT_ID`` and
``EUIPO_CLIENT_SECRET``). The client-credentials OAuth2 flow is enough: a search
returns pages of marks (RSQL ``query``, e.g. ``markFeature==FIGURATIVE``), the detail
of a mark carries ``markImage.viennaClasses`` (codes as ``01.01.02``), and
``/trademarks/{number}/image`` serves the picture. Rate limits are announced in the
``X-RateLimit-*`` headers and honoured with ``Retry-After``.

Marks examined since 2026-01-01 are coded with edition 10; the detail record does
not say which edition was used, so the application date is kept as a hint.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..config import EDITIONS
from ..scheme.codes import normalize_code
from .cases import EvalCase, images_dir

API_URL = "https://api.euipo.europa.eu/trademark-search"
TOKEN_URL = "https://euipo.europa.eu/cas-server-webapp/oidc/accessToken"


class EuipoClient:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        *,
        timeout: float = 60.0,
    ):
        self.client_id = client_id or os.environ.get("EUIPO_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("EUIPO_CLIENT_SECRET", "")
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "EUIPO credentials missing: set EUIPO_CLIENT_ID and EUIPO_CLIENT_SECRET "
                "(free developer account at https://dev.euipo.europa.eu)"
            )
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)
        self._token: str | None = None
        self._token_expires = 0.0

    def _headers(self) -> dict[str, str]:
        if self._token is None or time.time() > self._token_expires - 30:
            resp = self._http.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials", "scope": "uid"},
                auth=(self.client_id, self.client_secret),
            )
            resp.raise_for_status()
            body = resp.json()
            self._token = body["access_token"]
            self._token_expires = time.time() + float(body.get("expires_in", 3600))
        return {"Authorization": f"Bearer {self._token}", "X-IBM-Client-Id": self.client_id}

    def _get(self, path: str, **params) -> httpx.Response:
        for _attempt in range(6):
            resp = self._http.get(f"{API_URL}{path}", params=params, headers=self._headers())
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", "5")))
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"EUIPO API kept rate-limiting {path}")

    def search(
        self, query: str, *, page: int = 0, size: int = 100, sort: str | None = None
    ) -> dict:
        params = {"query": query, "page": page, "size": size}
        if sort:
            params["sort"] = sort
        return self._get("/trademarks", **params).json()

    def detail(self, application_number: str) -> dict:
        return self._get(f"/trademarks/{application_number}").json()

    def image(self, application_number: str) -> tuple[bytes, str]:
        resp = self._get(f"/trademarks/{application_number}/image")
        return resp.content, resp.headers.get("content-type", "")


def build_euipo_cases(
    n: int = 300,
    *,
    query: str = "markFeature==FIGURATIVE",
    sort: str | None = "applicationDate:desc",
    root: Path | None = None,
    client: EuipoClient | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[EvalCase]:
    """Fetch ``n`` figurative marks with Vienna codes and their images."""
    client = client or EuipoClient()
    out_dir = images_dir(root) / "euipo"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[EvalCase] = []
    page = 0
    while len(cases) < n:
        result = client.search(query, page=page, size=100, sort=sort)
        items = result.get("trademarks") or result.get("content") or []
        if not items:
            break
        for item in items:
            number = item["applicationNumber"]
            detail = client.detail(number)
            image = detail.get("markImage") or {}
            raw_codes = image.get("viennaClasses") or []
            codes = []
            for raw in raw_codes:
                try:
                    codes.append(normalize_code(raw))
                except ValueError:
                    continue
            if not codes:
                continue
            data, content_type = client.image(number)
            suffix = (
                ".png" if "png" in content_type else ".tif" if "tif" in content_type else ".jpg"
            )
            target = out_dir / f"{number}{suffix}"
            target.write_bytes(data)
            app_date = detail.get("applicationDate") or item.get("applicationDate")
            verbal = (detail.get("wordMarkSpecification") or {}).get("verbalElement")
            cases.append(
                EvalCase(
                    id=f"euipo:{number}",
                    image=str(target.relative_to(images_dir(root))),
                    vienna=tuple(dict.fromkeys(codes)),
                    edition=_edition_for(app_date),
                    text=verbal,
                    source=f"EUIPO Trademark Search API, filed {app_date}",
                    tags=("euipo", str(detail.get("markFeature", "")).lower()),
                )
            )
            if progress:
                progress(len(cases), n)
            if len(cases) >= n:
                break
        page += 1
    return cases


def _edition_for(application_date: str | None) -> str | None:
    """The edition in force on a date (an approximation of the one the examiner used)."""
    if not application_date:
        return None
    year = application_date[:4]
    chosen = None
    for edition, start in EDITIONS.items():
        if year >= start:
            chosen = edition
    return chosen
