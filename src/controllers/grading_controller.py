import logging
from fastapi import HTTPException
from controllers.BaseController import BaseController
from models.submission import GradingSubmission
from models.grading_result import GradingResult
from chains.grading_chain import GradingChain

logger = logging.getLogger(__name__)

class GradingController(BaseController):
    def __init__(self):
        super().__init__()
        self.logger = logger
        self.grading_chain = GradingChain()
        
    async def process_grading(self, submission: GradingSubmission) -> GradingResult:
        try:
            result = await self.grading_chain.run(submission)
            return result
        except Exception as e:
            self.logger.error(f"Error processing grading: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while grading the submission.")
