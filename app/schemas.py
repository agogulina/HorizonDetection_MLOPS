from typing import Optional
from pydantic import BaseModel, Field


class PredictMetaResponse(BaseModel):
    """
    JSON metadata returned alongside the mask image
    when calling POST /api/v1/predict/meta.
    """

    filename: str = Field(..., description="Original uploaded filename")
    mask_shape: list[int] = Field(
        ..., description="[height, width, num_classes] of the binary mask"
    )
    horizon_detected: bool = Field(
        ...,
        description=(
            "True when the model finds a clear sky/land boundary "
            "(i.e. both classes are present in the mask)"
        ),
    )
    roll_deg: Optional[float] = Field(
        None,
        description=(
            "Estimated roll angle in degrees. "
            "Positive = right wing down. None if horizon not detected."
        ),
    )
    pitch_deg: Optional[float] = Field(
        None,
        description=(
            "Estimated pitch angle in degrees. "
            "Positive = nose up. None if horizon not detected."
        ),
    )
    sky_ratio: float = Field(
        ..., description="Fraction of pixels classified as sky [0..1]"
    )
    land_ratio: float = Field(
        ..., description="Fraction of pixels classified as land [0..1]"
    )


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
