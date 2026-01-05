from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any

class EmailIn(BaseModel):
    sender_email: EmailStr
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=20000)

class TraceStep(BaseModel):
    step: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    reasoning: str

class OfferOption(BaseModel):
    option_name: str
    total_price: float
    discount_percent: float
    compliant: bool
    evidence: List[str] = Field(default_factory=list)
    reasoning: str

class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    purpose: str
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str

class ChatOut(BaseModel):
    category: str
    route: str
    intent: str
    summary: str
    recommendations: List[ProductRecommendation] = Field(default_factory=list)
    offers: List[OfferOption] = Field(default_factory=list)
    drafted_email: Optional[str] = None
    crm_opportunity: Optional[Dict[str, Any]] = None
    trace: List[TraceStep]
    conversation_id: Optional[str] = None
    assistant_message: Optional[str] = None
    customer_name: Optional[str] = None
    product_name: Optional[str] = None
    purchase_order: Optional[str] = None
    articleDoi: Optional[str] = None
