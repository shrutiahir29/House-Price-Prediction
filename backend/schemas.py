"""
HouseWise API — Data Schemas & Validation Rules
Defines the input contract for property features and output formats.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

# Allowed Categorical Values matching dataset
CityType = Literal[
    "Chandigarh",
    "Kolkata",
    "Delhi",
    "Ahmedabad",
    "Pune",
    "Hyderabad",
    "Mumbai",
    "Bangalore",
    "Jaipur",
    "Chennai",
]

LocalityType = Literal["Mid-Range", "Premium", "Affordable"]

PropertyType = Literal[
    "Builder Floor",
    "Villa",
    "Apartment",
    "Independent House",
]

FurnishingType = Literal["Semi-Furnished", "Unfurnished", "Furnished"]

ModelChoice = Literal["random_forest", "linear_regression", "xgboost", "decision_tree"]


class HouseInput(BaseModel):
    """Input payload containing all 16 property features for price prediction."""
    BHK: int = Field(..., ge=1, le=15, description="Number of bedrooms (BHK)", examples=[3])
    Bathrooms: int = Field(..., ge=1, le=15, description="Number of bathrooms", examples=[2])
    Super_Area_SqFt: float = Field(..., gt=100, le=20000, description="Super Built-up Area in SqFt", examples=[1500.0])
    Carpet_Area_SqFt: float = Field(..., gt=50, le=20000, description="Carpet Area in SqFt", examples=[1100.0])
    Floor_Number: int = Field(..., ge=0, le=100, description="Floor number of the house", examples=[5])
    Total_Floors: int = Field(..., ge=0, le=100, description="Total floors in the building", examples=[12])
    Age_of_Property: int = Field(..., ge=0, le=100, description="Age of property in years", examples=[5])
    Parking: int = Field(..., ge=0, le=10, description="Number of parking spaces", examples=[1])
    Lift_Available: int = Field(..., ge=0, le=1, description="1 if lift is available, 0 otherwise", examples=[1])
    Gated_Community: int = Field(..., ge=0, le=1, description="1 if gated society, 0 otherwise", examples=[1])
    Distance_to_Metro_km: float = Field(..., ge=0.0, le=50.0, description="Distance to nearest metro in km", examples=[2.5])
    Distance_to_City_Center_km: float = Field(..., ge=0.0, le=100.0, description="Distance to city center in km", examples=[8.0])
    
    City: CityType = Field(..., description="Indian Metropolitan City", examples=["Mumbai"])
    Locality_Type: LocalityType = Field(..., description="Locality tier", examples=["Premium"])
    Property_Type: PropertyType = Field(..., description="Type of property", examples=["Apartment"])
    Furnishing_Status: FurnishingType = Field(..., description="Furnishing level", examples=["Semi-Furnished"])

    model_name: Optional[ModelChoice] = Field(
        default="random_forest",
        description="ML algorithm to use ('random_forest' or 'linear_regression')"
    )

    @model_validator(mode="after")
    def validate_domain_rules(self):
        if self.Carpet_Area_SqFt > self.Super_Area_SqFt:
            raise ValueError("Carpet_Area_SqFt cannot exceed Super_Area_SqFt")
        if self.Floor_Number > self.Total_Floors:
            raise ValueError("Floor_Number cannot exceed Total_Floors")
        return self


class PredictionResponse(BaseModel):
    """Output response returned after property price prediction."""
    predicted_price_lakhs: float
    formatted_price: str
    model_used: str
    mode: Literal["real", "mock"]
    features_received: dict
    note: Optional[str] = None


class ModelMetricItem(BaseModel):
    """Evaluation metrics for a single ML algorithm."""
    model_id: str
    model_name: str
    mae: float
    rmse: float
    r2: float
    train_mae: Optional[float] = None
    train_rmse: Optional[float] = None
    train_r2: Optional[float] = None
    status: str
    is_best: bool = False


class MetricsResponse(BaseModel):
    """Comparative evaluation metrics across all four ML models."""
    project: str = "HouseWise"
    target_unit: str = "INR Lakhs"
    models: List[ModelMetricItem]
    best_model_id: str
    best_model_name: str
    summary: str
