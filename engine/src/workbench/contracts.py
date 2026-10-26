from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)
    qualification: str = ''


class ResearchOutput(BaseModel):
    headline: str
    summary: str
    findings: list[Finding]
    limitations: list[str]


class Calculation(BaseModel):
    label: str
    earlier: Decimal
    later: Decimal
    unit: str
    earlier_period: str
    later_period: str
    source_ids: list[str] = Field(min_length=1)

    def result(self):
        if self.earlier == 0:
            raise ValueError('Percentage change is undefined for a zero baseline')
        return {'absolute_change': str(self.later - self.earlier), 'percentage_change': str(((self.later - self.earlier) / abs(self.earlier) * 100).quantize(Decimal('.01'))), 'unit': self.unit, 'source_ids': self.source_ids}
