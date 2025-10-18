"""
dcn_client.py — Thin client for your DCN HTTP API + `dcn` SDK execution.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import requests
from eth_account import Account

class DCNClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url: str = base_url
        self.timeout: float = float(timeout)
        self.session: requests.Session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None

    # ---------- internals ----------
    def _handle_response(self, r: requests.Response):
        try:
            data = r.json()
        except Exception:
            r.raise_for_status()
            return {"raw": r.text}
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code} {data}", response=r)
        return data

    def _authz_headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        if self.refresh_token:
            h["X-Refresh-Token"] = self.refresh_token
        return h

    def _authorized_post_json(self, path: str, payload: Dict) -> Dict:
        url = f"{self.base_url}/{path}"
        r = self.session.post(url, json=payload, headers=self._authz_headers(), timeout=self.timeout)
        if r.status_code == 401 and self.refresh_token:
            self.refresh_tokens()
            r = self.session.post(url, json=payload, headers=self._authz_headers(), timeout=self.timeout)
        return self._handle_response(r)

    # ---------- public HTTP ----------
    def get_nonce(self, address: str) -> str:
        url = f"{self.base_url}/nonce/{address}"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        js = r.json()
        if "nonce" not in js:
            raise ValueError(f"Unexpected nonce response: {js}")
        return str(js["nonce"])

    def post_auth(self, address: str, message: str, signature: str) -> Dict:
        url = f"{self.base_url}/auth"
        r = self.session.post(url, json={"address": address, "message": message, "signature": signature}, timeout=self.timeout)
        data = self._handle_response(r)
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    def refresh_tokens(self) -> Dict:
        url = f"{self.base_url}/refresh"
        r = self.session.post(url, json={}, headers=self._authz_headers(), timeout=self.timeout)
        data = self._handle_response(r)
        self.access_token = data.get("access_token", self.access_token)
        self.refresh_token = data.get("refresh_token", self.refresh_token)
        return data

    def post_feature(self, payload: Dict) -> Dict:
        return self._authorized_post_json("feature", payload)

    def get_feature(self, name: str) -> Dict:
        """Optional: GET a feature by name if your API supports it."""
        url = f"{self.base_url}/feature/{name}"
        r = self.session.get(url, headers=self._authz_headers(), timeout=self.timeout)
        return self._handle_response(r)

    # ---------- `dcn` SDK execute ----------
    def execute_pt(self, acct: Account, pt_name: str, N: int,
                   seeds: Dict[str, int], dims: List[dict]) -> List[dict]:
        import dcn  # imported here so the module is optional unless used

        running = [(0, 0)] * (1 + len(dims))
        running[0] = (0, 0)
        for i, d in enumerate(dims):
            fname = (d.get("feature_name") or "").strip().lower()
            sp = int(seeds.get(fname, 0))
            running[i + 1] = (sp, 0)

        client = dcn.Client()
        try:
            client.login_with_account(acct)
            result = client.execute(pt_name, int(N), running)
            out: List[dict] = []
            for s in result:
                try:
                    out.append({"feature_path": s.feature_path, "data": list(s.data)})
                except Exception:
                    out.append({"feature_path": s.get("feature_path", ""), "data": list(s.get("data", []))})
            return out
        finally:
            try:
                client.close()
            except Exception:
                pass
