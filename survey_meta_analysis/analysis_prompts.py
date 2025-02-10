from typing import Dict
from schema import PersonaType

class AnalysisPrompts:
    """
    Collection of prompts for analyzing survey responses from both intel product reviewers and intel employees.
    Each analysis type (alignment, consistency, demographics) has specialized prompts for each persona type.
    """

    @staticmethod
    def get_alignment_prompt(persona_type: str, response_data: str, distribution_data: str) -> str:
        """Get the appropriate alignment analysis prompt based on persona type."""
        base_prompt = (
            AnalysisPrompts._intel_product_alignment_prompt() if persona_type == PersonaType.INTEL_PRODUCT_REVIEWER 
            else AnalysisPrompts._intel_employee_alignment_prompt()
        )
        return f"""
        {base_prompt}

        Survey Questions:
        {distribution_data}

        Response Data:
        {response_data}

        Return a JSON object with:
        {{
            "most_aligned_group": {{
                "group_name": string,
                "alignment_score": number,
                "member_count": number,
                "key_characteristics": List[string]
            }},
            "alignment_patterns": [
                {{
                    "group": string,
                    "score": number (0-1),
                    "size": number,
                    "common_traits": List[string]
                }}
            ],
            "notable_outliers": [
                {{
                    "description": string,
                    "reason": string
                }}
            ]
        }}
        """

    @staticmethod
    def get_consistency_prompt(persona_type: str, response_data: str, distribution_data: str) -> str:
        """Get the appropriate consistency analysis prompt based on persona type."""
        base_prompt = (
            AnalysisPrompts._intel_product_consistency_prompt() if persona_type == PersonaType.INTEL_PRODUCT_REVIEWER 
            else AnalysisPrompts._intel_employee_consistency_prompt()
        )
        return f"""
        {base_prompt}

        Survey Questions:
        {distribution_data}

        Response Data:
        {response_data}

        Return a JSON object with:
        {{
            "overall_consistency": {{
                "score": number (0-1),
                "confidence_level": number (0-1),
                "influential_factors": List[string]
            }},
            "consistency_by_group": [
                {{
                    "group": string,
                    "consistency_score": number (0-1),
                    "pattern_description": string
                }}
            ],
            "response_trends": [
                {{
                    "trend_description": string,
                    "affected_groups": List[string],
                    "significance": number (0-1)
                }}
            ]
        }}
        """

    @staticmethod
    def get_demographic_prompt(persona_type: str, response_data: str, distribution_data: str) -> str:
        """Get the appropriate demographic analysis prompt based on persona type."""
        base_prompt = (
            AnalysisPrompts._intel_product_demographic_prompt() if persona_type == PersonaType.INTEL_PRODUCT_REVIEWER 
            else AnalysisPrompts._intel_employee_demographic_prompt()
        )
        return f"""
        {base_prompt}

        Survey Questions:
        {distribution_data}

        Response Data:
        {response_data}

        Return a JSON object with:
        {{
            "role_based_insights": [
                {{
                    "role_type": string,
                    "key_patterns": List[string],
                    "sentiment_score": number (0-1)
                }}
            ],
            "experience_level_insights": [
                {{
                    "level": string,
                    "typical_responses": string,
                    "significant_differences": List[string]
                }}
            ],
            "demographic_correlations": [
                {{
                    "factor": string,
                    "correlation_strength": number (0-1),
                    "description": string
                }}
            ]
        }}
        """

    @staticmethod
    def _intel_product_alignment_prompt() -> str:
        """Product-specific prompt for alignment analysis."""
        return """
        Analyze how different types of product users align in their responses to feature preferences 
        and product satisfaction. Focus on identifying patterns based on use cases and technical requirements.

        Consider these user aspects:
        1. Technical expertise level (beginner to expert)
        2. Primary use cases (e.g., gaming, productivity, professional)
        3. Feature priorities and requirements
        4. Performance expectations
        5. Value perception and price sensitivity
        6. Integration needs and ecosystem compatibility
        7. Support requirements and expectations
        8. Update/maintenance preferences

        Look for alignments in:
        - Feature importance rankings
        - Performance requirements
        - Quality vs price tradeoffs
        - Technical sophistication needs
        - Problem-solving priorities
        - Integration requirements
        - Support expectations
        - Update/maintenance preferences

        Identify user groups based on:
        - Similar use case requirements
        - Technical expertise levels
        - Feature priority patterns
        - Value assessment approaches
        - Problem-solving needs
        - Integration complexity preferences
        """

    @staticmethod
    def _intel_employee_alignment_prompt() -> str:
        """Employee-specific prompt for alignment analysis."""
        return """
        Analyze the alignment patterns between different employee personas in their survey responses.
        Focus on identifying groups based on role, experience level, and workplace priorities.

        Consider these employee aspects:
        1. Job role and department
        2. Experience level and tenure
        3. Management level
        4. Work satisfaction factors
        5. Career development priorities
        6. Work-life balance needs
        7. Team collaboration preferences
        8. Company culture alignment

        Look for alignments in:
        - Job satisfaction factors
        - Management perceptions
        - Career growth expectations
        - Work-life balance priorities
        - Team dynamics preferences
        - Company culture views
        - Professional development needs
        - Workplace environment preferences

        Identify employee groups based on:
        - Similar role responsibilities
        - Experience level patterns
        - Management perspectives
        - Career development needs
        - Work style preferences
        - Cultural alignment patterns
        """

    @staticmethod
    def _intel_product_consistency_prompt() -> str:
        """Product-specific prompt for consistency analysis."""
        return """
        Analyze the consistency of product reviews and feedback across different aspects of the product.
        Focus on how opinions evolve with usage experience and across different features.

        Track consistency in these areas:
        1. Feature satisfaction over time
        2. Performance expectations vs reality
        3. Value perception stability
        4. Technical issue impacts
        5. Support experience influence
        6. Update/maintenance effects
        7. Integration satisfaction
        8. Ecosystem compatibility

        Consider these factors:
        - Usage duration impact
        - Technical expertise influence
        - Use case complexity
        - Problem resolution experience
        - Feature interaction patterns
        - Support interaction effects
        - Update/patch experiences
        - Integration challenges

        Look for patterns in:
        - Initial vs long-term impressions
        - Feature satisfaction stability
        - Performance assessment consistency
        - Value perception changes
        - Technical issue tolerance
        - Support satisfaction trends
        - Update impact patterns
        """

    @staticmethod
    def _intel_employee_consistency_prompt() -> str:
        """Employee-specific prompt for consistency analysis."""
        return """
        Analyze the consistency of responses across different questions for each employee persona.
        Focus on how views remain stable or change across various workplace aspects.

        Track consistency in these areas:
        1. Overall job satisfaction
        2. Management perception
        3. Career development views
        4. Work-life balance assessment
        5. Team dynamics
        6. Company culture
        7. Professional growth
        8. Workplace environment

        Consider these factors:
        - Role influence on views
        - Experience level impact
        - Management layer effects
        - Team size influence
        - Department culture
        - Location differences
        - Career stage
        - Work arrangement type

        Look for patterns in:
        - Satisfaction trend stability
        - Management view consistency
        - Career outlook changes
        - Work-life balance perception
        - Team collaboration views
        - Cultural alignment trends
        """

    @staticmethod
    def _intel_product_demographic_prompt() -> str:
        """Product-specific prompt for demographic analysis."""
        return """
        Analyze how different user segments respond to product features and capabilities.
        Focus on patterns based on technical expertise, use cases, and requirements.

        Analyze these user segments:
        1. Technical proficiency levels
        - Beginner users
        - Intermediate users
        - Advanced/power users
        - Professional/enterprise users

        2. Use case categories
        - Personal/home use
        - Professional/work use
        - Educational/academic use
        - Enterprise/organizational use

        3. Technical requirements
        - Basic functionality users
        - Performance-focused users
        - Integration-dependent users
        - Customization-heavy users

        4. Value perception groups
        - Budget-conscious users
        - Premium feature users
        - Enterprise/volume users
        - Specialized need users

        Consider these factors:
        - Feature priority patterns
        - Performance expectations
        - Support needs
        - Integration requirements
        - Update preferences
        - Customization needs
        - Price sensitivity
        - Technical limitations
        """

    @staticmethod
    def _intel_employee_demographic_prompt() -> str:
        """Employee-specific prompt for demographic analysis."""
        return """
        Analyze how different demographic and role-based groups respond to workplace factors.
        Focus on patterns based on experience level, role type, and other demographics.

        Analyze these employee segments:
        1. Role categories
        - Individual contributors
        - Team leads/managers
        - Department heads
        - Executive level

        2. Experience levels
        - Entry level
        - Mid-career
        - Senior level
        - Leadership

        3. Department types
        - Technical roles
        - Business functions
        - Support services
        - Administrative roles

        4. Location factors
        - Office-based
        - Remote workers
        - Hybrid arrangement
        - Multiple locations

        Consider these factors:
        - Career progression patterns
        - Management relationships
        - Work-life balance needs
        - Professional development
        - Team collaboration
        - Cultural alignment
        - Workplace flexibility
        - Resource access
        """