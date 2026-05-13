import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import CVE, CvssMetric, NVDResponse

logger = logging.getLogger("vuln_loader")
logging.basicConfig(level=logging.INFO)

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = os.getenv("NVD_API_KEY")
FAISS_URL = os.getenv("FAISS_URL", "http://faiss:8000")
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://lightrag:9621")

DEFAULT_CVE_IDS = [
    "CVE-2021-44228", "CVE-2021-45046", "CVE-2017-5638", "CVE-2021-41773",
    "CVE-2022-22965", "CVE-2022-22963", "CVE-2014-0160", "CVE-2022-0778",
    "CVE-2022-3602", "CVE-2022-3786", "CVE-2019-5736", "CVE-2022-0185",
    "CVE-2021-25741", "CVE-2018-15664", "CVE-2023-44487", "CVE-2021-26855",
    "CVE-2022-1388", "CVE-2021-34527", "CVE-2023-23397", "CVE-2021-40539",
]



async def fetch_cve(client: httpx.AsyncClient, cve_id: str) -> CVE | None:
    params = {"cveId": cve_id}
    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    resp = await client.get(NVD_BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    nvd = NVDResponse.model_validate(resp.json())
    return nvd.vulnerabilities[0].cve if nvd.vulnerabilities else None



def _best_cvss(cve: CVE) -> CvssMetric | None:
    if not cve.metrics:
        return None
    for field in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metrics = getattr(cve.metrics, field, None)
        if metrics:
            return metrics[0]
    return None


def cve_to_text(cve: CVE) -> str:

    desc = next((desc.value for desc in cve.descriptions if desc.lang == "en"), "")
    cwes = []
    for weakness in cve.weaknesses or []:
        for wd in weakness.description:
            if wd.lang == "en" and wd.value != "NVD-CWE-noinfo":
                cwes.append(wd.value)

    metric = _best_cvss(cve)
    severity = "N/A"
    score = "N/A"
    attack_vector = "N/A"
    if metric:
        score = str(metric.cvssData.baseScore)
        severity = metric.cvssData.baseSeverity or metric.baseSeverity or "N/A"
        attack_vector = (
            metric.cvssData.attackVector
            or metric.cvssData.accessVector
            or "N/A"
        )

    refs = [r.url for r in (cve.references or [])[:5]]

    lines = [
        f"CVE ID: {cve.id}",
        f"Published: {cve.published}",
        f"Status: {cve.vulnStatus or 'Unknown'}",
        f"Severity: {severity} (CVSS {score})",
        f"Attack Vector: {attack_vector}",
        f"CWE: {', '.join(cwes) if cwes else 'N/A'}",
        f"Description: {desc}",
    ]
    if refs:
        lines.append(f"References: {', '.join(refs)}")

    return "\n".join(lines)



def _cve_metadata(cve_id: str, cve: CVE) -> dict:
    metric = _best_cvss(cve)
    severity = "unknown"
    score = "N/A"
    attack_vector = "N/A"
    if metric:
        score = str(metric.cvssData.baseScore)
        severity = metric.cvssData.baseSeverity or metric.baseSeverity or "unknown"
        attack_vector = (
            metric.cvssData.attackVector
            or metric.cvssData.accessVector
            or "N/A"
        )

    cwes = []
    for weakness in cve.weaknesses or []:
        for wd in weakness.description:
            if wd.lang == "en" and wd.value != "NVD-CWE-noinfo":
                cwes.append(wd.value)

    return {
        "cve_id": cve_id,
        "published": cve.published,
        "status": cve.vulnStatus or "Unknown",
        "severity": severity,
        "cvss_score": score,
        "attack_vector": attack_vector,
        "cwe": ", ".join(cwes) if cwes else "N/A",
        "references": [r.url for r in (cve.references or [])[:5]],
    }


async def sync_to_faiss(client: httpx.AsyncClient, cves: dict[str, CVE]) -> dict:
    documents = []
    for cve_id, cve in cves.items():
        desc = next(
            (d.value for d in cve.descriptions if d.lang == "en"), ""
        )
        documents.append({
            "source_id": cve_id,
            "content": desc,
            "metadata": _cve_metadata(cve_id, cve),
        })
    resp = await client.post(
        f"{FAISS_URL}/sync",
        json={"documents": documents, "delete_missing": False},
    )
    resp.raise_for_status()
    return resp.json()


async def sync_to_lightrag(client: httpx.AsyncClient, cves: dict[str, CVE]) -> dict:
    texts = [cve_to_text(cve) for cve in cves.values()]
    file_sources = list(cves.keys())
    resp = await client.post(
        f"{LIGHTRAG_URL}/documents/texts",
        json={"texts": texts, "file_sources": file_sources},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_cves_from_nvd(
    client: httpx.AsyncClient, cve_ids: list[str]
) -> tuple[dict[str, CVE], list[str]]:
    cves: dict[str, CVE] = {}
    failed: list[str] = []
    for cve_id in cve_ids:
        cve_id = cve_id.upper()
        try:
            cve = await fetch_cve(client, cve_id)
            if cve:
                cves[cve.id] = cve
            else:
                failed.append(cve_id)
        except Exception as e:
            logger.error(f"Failed to fetch {cve_id}: {e}")
            failed.append(cve_id)
    return cves, failed


class LoadRequest(BaseModel):
    cve_ids: list[str]


class LoadResponse(BaseModel):
    fetched: list[str]
    failed: list[str]
    faiss: dict | None = None
    lightrag: dict | None = None
    errors: list[str]


app = FastAPI(title="Vuln Loader", description="CVE fetch & sync pipeline")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cves/{cve_id}")
async def get_cve(cve_id: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        cve = await fetch_cve(client, cve_id.upper())
    if not cve:
        raise HTTPException(status_code=404, detail=f"{cve_id} not found in NVD")
    return cve.model_dump()


@app.post("/load", response_model=LoadResponse)
async def load_cves(request: LoadRequest):
    async with httpx.AsyncClient(timeout=30.0) as client:
        cves, failed = await fetch_cves_from_nvd(client, request.cve_ids)

    if not cves:
        return LoadResponse(fetched=[], failed=failed, errors=["No CVEs fetched"])

    errors: list[str] = []
    faiss_result = None
    lightrag_result = None

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            faiss_result = await sync_to_faiss(client, cves)
        except Exception as e:
            logger.error(f"FAISS sync failed: {e}")
            errors.append(f"faiss: {e}")

        try:
            lightrag_result = await sync_to_lightrag(client, cves)
        except Exception as e:
            logger.error(f"LightRAG sync failed: {e}")
            errors.append(f"lightrag: {e}")

    return LoadResponse(
        fetched=sorted(cves.keys()),
        failed=failed,
        faiss=faiss_result,
        lightrag=lightrag_result,
        errors=errors,
    )


@app.post("/load/defaults", response_model=LoadResponse)
async def load_defaults():
    return await load_cves(LoadRequest(cve_ids=DEFAULT_CVE_IDS))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
