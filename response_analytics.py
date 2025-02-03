# survey_analytics.py
from typing import List, Dict, Any
import numpy as np
import statsmodels.stats.proportion as smp
from question_classifier import QuestionClassifier
from anthropic import AsyncAnthropicBedrock
import json
from qualitative_analytics import QuestionQualitativeAnalysis
import time
class QuestionAnalytics:
    def __init__(self, all_responses: List[Dict[str, float]], n_samples: int = 2000):
        """
        Initialize QuestionAnalytics with the list of persona responses

        Args:
            all_responses: List[Dict[str, float]] - List of probability distributions from LLM
            n_samples: int - Number of samples to generate per distribution
        """
        self.valid_responses = [resp for resp in all_responses if not resp.get('error')]
        self.responses = [resp.get('distribution') for resp in self.valid_responses]
        self.n_samples = n_samples
        self.combined_samples = None
    
        self.reliability_scores = [
            resp.get('reliability_score', 0) 
            for resp in self.valid_responses 
            if resp.get('reliability_score') is not None
        ]

        self.question_classifier = QuestionClassifier()
        self.qualitative_analysis = QuestionQualitativeAnalysis(self.valid_responses) # valid responses contain the persona_id and other data


    def generate_samples(self) -> np.ndarray:
        """Generate samples from multiple distributions"""        
        # Pre-allocate the array for all samples
        all_samples = np.empty((len(self.responses), self.n_samples), dtype=object)
        
        for i, dist in enumerate(self.responses):
            options = np.array(list(dist.keys()))
            probs = np.array([dist[opt] for opt in options])
            
            # Add validation
            total = np.sum(probs)
            if not np.isclose(total, 1.0, rtol=1e-5):
                print(f"[QuestionAnalytics][generate_samples] Distribution {i} sums to {total}: {dict(zip(options, probs))}")
                # Normalize the probabilities
                probs = probs / total
            
            all_samples[i] = np.random.choice(options, size=self.n_samples, p=probs)
        
        return all_samples.flatten()

    def calculate_mean_reliability(self) -> float:
        """Calculate mean reliability score across all valid responses"""
        if not self.reliability_scores:
            return 0.0
        return float(np.mean(self.reliability_scores))

    def calculate_basic_stats(self) -> Dict:
        """Calculate basic statistics including counts and percentages"""
        if self.combined_samples is None:
            self.combined_samples = self.generate_samples()

        unique, counts = np.unique(self.combined_samples, return_counts=True)
        total = len(self.combined_samples)
        
        # Convert numpy types to Python native types
        stats_dict = {
            "frequencies": {str(k): int(v) for k, v in zip(unique, counts)},
            "proportions": {str(k): float(v) for k, v in zip(unique, counts/total)},
            "percentages": {str(k): float(v) for k, v in zip(unique, (counts/total) * 100)},
            "total_responses": int(total),
            "confidence_intervals": {}
        }
        
        # Calculate Wilson confidence intervals
        for opt, count in zip(unique, counts):
            lower, upper = smp.proportion_confint(
                count, total, alpha=0.05, method='wilson'
            )
            stats_dict["confidence_intervals"][str(opt)] = {
                "lower": float(lower * 100),
                "upper": float(upper * 100)
            }
            
        return stats_dict

    def calculate_categorical_metrics(self) -> Dict:
        """Calculate metrics appropriate for categorical data"""
        stats = self.calculate_basic_stats()
        
        # Find mode (most common response)
        mode = max(stats["frequencies"].items(), key=lambda x: x[1])
        
        # Calculate diversity of responses
        props = list(stats["proportions"].values())
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in props)
        
        return {
            "most_common_response": {
                "option": mode[0],
                "frequency": mode[1],
                "proportion": stats["proportions"][mode[0]]
            },
            "response_entropy": entropy,  # Higher means more diverse responses
            "number_of_options_chosen": len(stats["frequencies"])
        }

    def calculate_agreement_metrics(self, ordered_options: List[str]) -> Dict:
        """Calculate agreement metrics for Likert scales"""
        if self.combined_samples is None:
            self.combined_samples = self.generate_samples()
            
        n_options = len(ordered_options)
        total_samples = len(self.combined_samples)
        
        # Handle different numbers of options
        if n_options < 3:
            # For binary/small scales, just use the last option as positive
            top_options = [ordered_options[-1]]
            bottom_options = [ordered_options[0]]
        else:
            # For larger scales, use top two and bottom two
            top_options = ordered_options[-2:]
            bottom_options = ordered_options[:2]
        
        top_count = sum(1 for s in self.combined_samples if s in top_options)
        bottom_count = sum(1 for s in self.combined_samples if s in bottom_options)
        
        top_box = (top_count / total_samples) * 100
        bottom_box = (bottom_count / total_samples) * 100
        
        return {
            "top_box_score": {
                "percentage": top_box,
                "count": top_count
            },
            "bottom_box_score": {
                "percentage": bottom_box,
                "count": bottom_count
            },
            "net_score": top_box - bottom_box,
            "options_used": {
                "top": top_options,
                "bottom": bottom_options
            }
        }

    def calculate_polarization(self, ordered_options: List[str]) -> Dict:
        """Calculate polarization metrics for Likert scales"""
        if self.combined_samples is None:
            self.combined_samples = self.generate_samples()
            
        n_options = len(ordered_options)
        
        # Create scale mapping from options to numeric values
        scale = {opt: i/(n_options-1) for i, opt in enumerate(ordered_options)}
        numeric_samples = [scale[s] for s in self.combined_samples]
        
        # Calculate distance from middle
        middle = 0.5
        distances = [abs(x - middle) for x in numeric_samples]
        
        return {
            "polarization_index": np.mean(distances),
            "extreme_response_rate": sum(1 for d in distances if d > 0.4) / len(distances)
        }

    async def question_classification(self, options: List[str]) -> Dict[str, Any]:
        """Classify the question type using LLM"""
        analysis = await self.question_classifier.classify(options)
        return analysis

    async def analyze_survey_question(self, question: str, options: List[str]) -> Dict[str, Any]:
        """Main method to analyze a survey question"""
        start_time = time.time()
        # First, analyze the question type using LLM
        analysis = await self.question_classifier.classify(question, options)
        print(f"-- Time taken to classify question: {time.time() - start_time}")
        
        # Calculate basic statistics
        basic_stats = self.calculate_basic_stats()
        print(f"-- Time taken to calculate basic stats: {time.time() - start_time}")
        mean_reliability = self.calculate_mean_reliability()
        print(f"-- Time taken to calculate mean reliability: {time.time() - start_time}")
        
        results = {
            "question_type": analysis["scale_type"],
            "basic_statistics": basic_stats,
            "mean_reliability": mean_reliability
        }
        
        # Add metrics based on question type
        if analysis["is_likert"] and analysis["ordered_options"]:
            results.update({
                "agreement_metrics": self.calculate_agreement_metrics(analysis["ordered_options"]),
                "polarization_metrics": self.calculate_polarization(analysis["ordered_options"]),
                "ordered_options": analysis["ordered_options"]
            })
            print(f"-- Time taken to calculate agreement metrics: {time.time() - start_time}")
        else:
            results.update({
                "categorical_metrics": self.calculate_categorical_metrics()
            })
            print(f"-- Time taken to calculate categorical metrics: {time.time() - start_time}")
        #qualititave_analysis
        qualitative_analysis = await self.qualitative_analysis.analyze_question(question, options)
        print(f"-- Time taken to qualitative analysis: {time.time() - start_time}")
        results.update({
            "theme_analysis": qualitative_analysis["theme_analysis"] if "theme_analysis" in qualitative_analysis else {},
            "network_analysis": qualitative_analysis["network_analysis"] if "network_analysis" in qualitative_analysis else {},
            "sentiment_analysis": qualitative_analysis["sentiment_analysis"] if "sentiment_analysis" in qualitative_analysis else {},
            "response_patterns": qualitative_analysis["response_patterns"] if "response_patterns" in qualitative_analysis else {}
        })

        return results

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_survey_analytics():
        # Example with Likert scale
        all_responses = [
            {
                'persona_id': '0',
                'personality_summary': "David Chen is a seasoned data scientist with over a decade of experience, currently based in Hillsboro, OR. He rates his company positively at 4.0 out of 5, highlighting the good pay and the camaraderie he enjoyed with his colleagues. David values autonomy in his work, appreciating the lack of micromanagement and the variety in his job duties, which kept him engaged.\n\nHowever, David's overall sentiment towards the company is mixed, as he expresses significant concerns about its management practices and business outlook. He is particularly critical of the frequent headcount reductions, which he believes disrupt continuity and waste resources on training new employees only to have them let go. This cycle of hiring and layoffs has led to frustration and a sense of instability, prompting him to leave the company after experiencing four such cycles.\n\nDavid is also discontent with the promotion process, feeling that it favors educational credentials over practical experience, which he believes should be prioritized. He perceives the company culture as inflexible, stifling innovation and lacking diversity in perspectives, as it predominantly hires recent graduates rather than experienced professionals from other organizations.\n\nIn summary, David Chen is a dedicated and experienced professional who values a supportive work environment and stability. While he appreciates certain aspects of his job, he is deeply concerned about the company's management practices, the impact of its hiring cycles, and the need for a more flexible and innovative culture.",
                'distribution': {
                    'Very Dissatisfied': 0.13333333333333333,
                    'Dissatisfied': 0.2333333333333333,
                    'Neutral': 0.21666666666666667,
                    'Satisfied': 0.26666666666666666,
                    'Very Satisfied': 0.15
                },
                'reason': 'The employee has mixed feelings about work, appreciating pay and autonomy but frustrated by long hours and management issues.',
                'error': None
            },
            {
                'persona_id': '1', 
                'personality_summary': "Michael Zhang is a seasoned Senior Component Design Engineer with over a decade of experience, currently based in Hillsboro, OR. He has a positive overall rating of 4.0 out of 5 for his company, reflecting a balanced view of his work environment. Michael appreciates the decent pay and benefits, as well as the intelligence and friendliness of his coworkers. He values Intel's long-standing history of achievement and recognizes the potential for career advancement within development groups, provided one is willing to invest time and effort.\n\nHowever, Michael's perspective is tempered by some significant concerns. He expresses dissatisfaction with the company's leadership, particularly the CEO, and holds a negative outlook on the business's future. His primary frustrations stem from organizational inefficiencies, such as poorly managed projects that lead to wasted time and excessive multitasking. He is particularly vocal about the need for uninterrupted time for deep work, emphasizing that employees engaged in complex tasks like architecture, design, or validation should not be overloaded with parallel assignments or burdened by non-productive meetings. Michael advocates for a more structured approach to meetings, insisting on clear agendas and the inclusion of only essential participants to ensure productive outcomes.\n\nOverall, Michael's personality reflects a pragmatic and thoughtful engineer who values efficiency, collaboration, and a conducive work environment, while also being critical of leadership and organizational practices that hinder productivity.",
                'distribution': {
                    'Very Dissatisfied': 0.14999999999999997,
                    'Dissatisfied': 0.24999999999999997,
                    'Neutral': 0.19999999999999996,
                    'Satisfied': 0.28333333333333327,
                    'Very Satisfied': 0.11666666666666665
                },
                'reason': 'The employee has mixed feelings about work-life balance, appreciating pay and coworkers but frustrated by meetings and multitasking challenges.',
                'error': None
            },
            {
                'persona_id': '2',
                'personality_summary': "Based on the provided attributes, Sarah Martinez can be summarized as a deeply dissatisfied and frustrated marketing leader who has significant concerns about her company's direction and leadership. With over five years of experience at the company, she has witnessed a decline in its performance and culture, leading her to rate it a mere 1.0 out of 5. \n\nSarah acknowledges some positive aspects of her job, such as the company's past reputation and the presence of some competent colleagues, but these are overshadowed by her strong disapproval of the current CEO and senior management. She perceives the CEO as incompetent and misleading, attributing the company's struggles to poor leadership decisions and a lack of strategic vision. Her sentiments reflect a broader disillusionment with the company's culture, compensation, and overall future prospects.\n\nHer key concerns center around the urgent need for a new CEO and a complete overhaul of the senior management team, indicating a desire for significant change to restore the company to its former glory. Overall, Sarah's personality reflects a critical, pragmatic, and somewhat cynical view of her workplace, driven by a strong desire for improvement and accountability in leadership.",
                'distribution': {
                    'Very Dissatisfied': 0.43333333333333335,
                    'Dissatisfied': 0.3499999999999999,
                    'Neutral': 0.15,
                    'Satisfied': 0.06,
                    'Very Satisfied': 0.006666666666666667
                },
                'reason': 'The employee feels severely dissatisfied due to poor leadership, lack of direction, and negative work environment impacting their work-life balance.',
                'error': None
            }
        ]
        
        question_analyzer = QuestionAnalytics(all_responses)
        
        likert_results = await question_analyzer.analyze_survey_question(
            "How satisfied are you with your work-life balance?",
            list(all_responses[0].get('distribution').keys())
        )
        print('Likert Results:', json.dumps(likert_results, indent=2))
    
    asyncio.run(test_survey_analytics())