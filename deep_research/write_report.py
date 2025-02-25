from rich.console import Console
from typing import List
from report_prompt import prompts_object
from langfuse.openai import AzureOpenAI
from providers import get_ai_client
import json
import asyncio


console = Console()
client = get_ai_client("azure", console)

from pydantic import BaseModel, Field
from typing import List, Dict


class MarketSizeIndustryAnalysis(BaseModel):
    total_addressable_market: str = Field(..., description="Concrete numbers and metrics for the Total Addressable Market (TAM)")
    market_segmentation: str = Field(..., description="Detailed market segmentation with clear data points")
    industry_growth: str = Field(..., description="Industry growth rates and future projections")
    key_market_drivers: str = Field(..., description="Key market drivers supported by specific evidence and data")


class BottomUpAnalysis(BaseModel):
    quantitative_analysis: str = Field(..., description="Detailed quantitative analysis of survey responses, including statistics")
    segmentation: str = Field(..., description="Clear segmentation based on response patterns")
    statistical_analysis: str = Field(..., description="Statistical analysis of response distributions")
    persona_insights: str = Field(..., description="Data-backed persona insights and quotes")


class CompetitiveLandscape(BaseModel):
    market_share_data: str = Field(..., description="Market share trends with specific numbers")
    competitor_mapping: str = Field(..., description="Detailed competitor product mapping")
    price_performance_analysis: str = Field(..., description="Price vs. performance analysis across segments")
    positioning_matrix: str = Field(..., description="Competitive positioning matrix")


class ProductPerformanceAnalysis(BaseModel):
    performance_metrics: str = Field(
        ..., 
        description="Key performance indicators and metrics specific to the product category (e.g., durability for apparel, taste ratings for food, processing speed for technology)"
    )
    comparative_analysis: str = Field(
        ..., 
        description="Detailed comparison of product performance against competitors across relevant dimensions (e.g., comfort vs price for apparel, flavor profiles for snacks, functionality for tech)"
    )
    value_proposition: str = Field(
        ..., 
        description="Analysis of price-to-value relationship, including consumer perceived benefits relative to cost, and positioning in different price segments"
    )
    product_attributes: str = Field(
        ..., 
        description="Comprehensive comparison of product characteristics (e.g., materials and style for apparel, ingredients and nutrition for snacks, features and specifications for tech products)"
    )


class IndustryTrendsFutureOutlook(BaseModel):
    technology_evolution: str = Field(..., description="Technology roadmap and evolution trends")
    market_growth_projections: str = Field(..., description="Future market growth projections with metrics")
    regulatory_considerations: str = Field(..., description="Relevant regulatory considerations impacting the industry")
    future_scenarios: str = Field(..., description="Analysis of potential future market scenarios")


class Visualizations(BaseModel):
    market_share_charts: str = Field(..., description="Structured data for market share charts")
    competitive_matrices: str = Field(..., description="Data for competitive positioning matrices")
    performance_tables: str = Field(..., description="Data for performance comparison tables")
    trend_graphs: str = Field(..., description="Data for trend analysis graphs")
    segment_charts: str = Field(..., description="Data for segment distribution charts")

class MarketResearchReport(BaseModel):
    name: str = Field(..., description="Name of the report")
    market_size_industry_analysis: MarketSizeIndustryAnalysis
    bottom_up_analysis: BottomUpAnalysis
    competitive_landscape: CompetitiveLandscape
    product_performance_analysis: ProductPerformanceAnalysis
    industry_trends_future_outlook: IndustryTrendsFutureOutlook
    actionable_insights: str = Field(..., description="Clear and concise strategic recommendations based on the analysis")
    visualizations: Visualizations
    citations: List[str] = Field(..., description="Inline citations referencing specific sources used in the report")


# Main Response Schema for API Compatibility
class ResponseFormat(BaseModel):
    name: str = Field(..., description="Name of the report")
    report_schema: MarketResearchReport






async def write_final_report(
    client: AzureOpenAI,
) -> str:
    """Generate final report based on all research learnings."""

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.beta.chat.completions.parse(
            model="o3-mini",
            max_completion_tokens=100000,
            messages=prompts_object,
            response_format=ResponseFormat,
        ),
    )

    try:
        result = json.loads(response.choices[0].message.content)
        print(f"[deep_research] [write_final_report] result: {result}")
        return result
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.choices[0].message.content}")
        return "Error generating report"

asyncio.run(write_final_report(client))