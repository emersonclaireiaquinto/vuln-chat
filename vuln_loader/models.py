from __future__ import annotations

from pydantic import BaseModel


class Description(BaseModel):
    lang: str
    value: str


class CvssData(BaseModel):
    version: str
    vectorString: str
    baseScore: float
    baseSeverity: str | None = None
    attackVector: str | None = None
    attackComplexity: str | None = None
    privilegesRequired: str | None = None
    userInteraction: str | None = None
    scope: str | None = None
    confidentialityImpact: str | None = None
    integrityImpact: str | None = None
    availabilityImpact: str | None = None
    accessVector: str | None = None
    accessComplexity: str | None = None
    authentication: str | None = None


class CvssMetric(BaseModel):
    source: str
    type: str
    cvssData: CvssData
    exploitabilityScore: float | None = None
    impactScore: float | None = None
    baseSeverity: str | None = None
    acInsufInfo: bool | None = None
    obtainAllPrivilege: bool | None = None
    obtainUserPrivilege: bool | None = None
    obtainOtherPrivilege: bool | None = None
    userInteractionRequired: bool | None = None


class Metrics(BaseModel):
    cvssMetricV40: list[CvssMetric] | None = None
    cvssMetricV31: list[CvssMetric] | None = None
    cvssMetricV30: list[CvssMetric] | None = None
    cvssMetricV2: list[CvssMetric] | None = None


class Weakness(BaseModel):
    source: str
    type: str
    description: list[Description]


class Reference(BaseModel):
    url: str
    source: str | None = None
    tags: list[str] | None = None


class CveTag(BaseModel):
    sourceIdentifier: str
    tags: list[str]


class CVE(BaseModel):
    id: str
    sourceIdentifier: str | None = None
    published: str
    lastModified: str
    vulnStatus: str | None = None
    cveTags: list[CveTag] | None = None
    descriptions: list[Description]
    metrics: Metrics | None = None
    weaknesses: list[Weakness] | None = None
    configurations: list[dict] | None = None
    references: list[Reference] | None = None
    evaluatorComment: str | None = None
    evaluatorSolution: str | None = None
    evaluatorImpact: str | None = None


class VulnerabilityItem(BaseModel):
    cve: CVE


class NVDResponse(BaseModel):
    resultsPerPage: int
    startIndex: int
    totalResults: int
    format: str
    version: str
    timestamp: str
    vulnerabilities: list[VulnerabilityItem]


class NVDRequest(BaseModel):
    cveId: str | None = None
    cpeName: str | None = None
    cveTag: str | None = None
    cweId: str | None = None
    sourceIdentifier: str | None = None
    keywordSearch: str | None = None
    keywordExactMatch: bool | None = None
    cvssV2Metrics: str | None = None
    cvssV3Metrics: str | None = None
    cvssV4Metrics: str | None = None
    cvssV2Severity: str | None = None
    cvssV3Severity: str | None = None
    cvssV4Severity: str | None = None
    isVulnerable: bool | None = None
    hasKev: bool | None = None
    hasCertAlerts: bool | None = None
    hasCertNotes: bool | None = None
    hasOval: bool | None = None
    pubStartDate: str | None = None
    pubEndDate: str | None = None
    lastModStartDate: str | None = None
    lastModEndDate: str | None = None
    kevStartDate: str | None = None
    kevEndDate: str | None = None
    startIndex: int | None = None
    resultsPerPage: int | None = None
