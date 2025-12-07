from app.apis.inferences_routes import router as inference_route
from fastapi import APIRouter, status
from fastapi.responses import Response

router = APIRouter()

# Define the API endpoints

router.include_router(inference_route)

# Health check endpoint
@router.get("/")
def health_check():
    return Response(
        content="AI Service is running", status_code=status.HTTP_200_OK
    )
