from fastapi import APIRouter, Depends

from controllers.grading_controller import GradingController
from models.grading_result import GradingResult
from models.submission import GradingSubmission

router = APIRouter(
    prefix="/api/grading",
    tags=["Grading"],
)


async def get_grading_controller():
    return GradingController()


@router.post("/submit", response_model=GradingResult)
async def submit_grading(
    submission: GradingSubmission,
    controller: GradingController = Depends(get_grading_controller),
):
    """Submits a student's answers for automated grading against expected criteria."""
    return await controller.process_grading(submission)
