from dotenv import load_dotenv
from anthropic import Anthropic
import json
import time
from typing import Dict
from pathlib import Path

class EarningsAnalyzer:
    def __init__(self):
        load_dotenv()
        self.client = Anthropic()
        self.model = "claude-3-5-sonnet-latest"
        self.temperature = 0.7
        
    def load_transcript(self, filename: str) -> str:
        """Load transcript from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('content', '')
                elif isinstance(data, dict):
                    return data.get('content', '')
                return ''
        except Exception as e:
            print(f"Error loading transcript: {str(e)}")
            return ''

    def generate_prompt(self, transcript: str, analysis_type: str) -> str:
        """Generate analysis prompts"""
        prompts = {
            "ai_initiatives": """
                Extract all AI-related information from the transcript.
                Return a JSON object with this exact structure:
                {
                    "core_strategy": {
                        "initiatives": [],
                        "timeline": [],
                        "investments": []
                    },
                    "features": {
                        "current": [],
                        "planned": [],
                        "technical_requirements": []
                    },
                    "partnerships": {
                        "announced": [],
                        "planned": []
                    },
                    "infrastructure": {
                        "compute": [],
                        "processing_approach": []
                    },
                    "business_impact": {
                        "revenue_potential": [],
                        "competitive_advantages": []
                    }
                }
                
                Include only factual information mentioned in the transcript.
                Each array should contain simple strings with specific facts.
            """
        }
        
        return f"""You are a technology analyst focusing on AI developments.
        Analyze this earnings transcript and provide specific, concrete information.
        Format your response as clean, properly structured JSON.
        
        {prompts.get(analysis_type, 'Analyze the transcript for key insights.')}
        
        Transcript:
        {transcript}
        """

    def analyze_transcript(self, transcript: str) -> Dict:
        """Analyze transcript content"""
        if not transcript:
            return {"error": "No transcript content provided"}
            
        results = {}
        analysis_types = ["ai_initiatives"]
        
        for analysis_type in analysis_types:
            print(f"\nAnalyzing {analysis_type}...")
            
            try:
                stream = self.client.messages.create(
                    messages=[{
                        "role": "user",
                        "content": self.generate_prompt(transcript, analysis_type)
                    }],
                    model=self.model,
                    max_tokens=4000,
                    temperature=self.temperature,
                    stream=True
                )
                
                response = ""
                for event in stream:
                    if event.type == "content_block_delta":
                        response += event.delta.text
                        print(event.delta.text, end="", flush=True)
                
                # Clean and parse JSON response
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:-3]
                if cleaned_response.startswith('{'):
                    results[analysis_type] = json.loads(cleaned_response)
                else:
                    results[analysis_type] = {"error": "Invalid JSON response"}
                
            except Exception as e:
                print(f"\nError in {analysis_type} analysis: {str(e)}")
                results[analysis_type] = {"error": str(e)}
            
            time.sleep(2)
        
        return results

    def generate_ai_summary(self, results: Dict) -> str:
        """Generate readable summary of AI findings"""
        ai_data = results.get('ai_initiatives', {})
        if isinstance(ai_data, dict) and not ai_data.get('error'):
            summary = "\nAI Initiatives Summary"
            summary += "\n==================="
            
            # Core Strategy
            if ai_data.get('core_strategy', {}).get('initiatives'):
                summary += "\n\nCore AI Strategy:"
                for item in ai_data['core_strategy']['initiatives']:
                    summary += f"\n• {item}"
            
            # Features
            if ai_data.get('features', {}).get('planned'):
                summary += "\n\nPlanned AI Features:"
                for item in ai_data['features']['planned']:
                    summary += f"\n• {item}"
            
            # Partnerships
            if ai_data.get('partnerships', {}).get('announced'):
                summary += "\n\nAI Partnerships:"
                for item in ai_data['partnerships']['announced']:
                    summary += f"\n• {item}"
            
            # Impact
            if ai_data.get('business_impact', {}).get('competitive_advantages'):
                summary += "\n\nBusiness Impact:"
                for item in ai_data['business_impact']['competitive_advantages']:
                    summary += f"\n• {item}"
            
            return summary
        else:
            return "\nError: Could not generate AI summary due to invalid data structure."

    def save_analysis(self, results: Dict, filename: str = "ai_analysis.json"):
        """Save analysis results to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nAnalysis saved to {filename}")
        except Exception as e:
            print(f"\nError saving analysis: {str(e)}")

def main():
    # Initialize analyzer
    analyzer = EarningsAnalyzer()
    
    # Load transcript
    transcript_file = "aapl_transcript_20241023.json"
    if not Path(transcript_file).exists():
        print(f"Error: Transcript file {transcript_file} not found!")
        return
    
    print(f"Loading transcript from {transcript_file}...")
    transcript = analyzer.load_transcript(transcript_file)
    
    if not transcript:
        print("Error: No transcript content found!")
        return
    
    # Run analysis
    print("Starting analysis...")
    results = analyzer.analyze_transcript(transcript)
    
    # Generate and print summary
    summary = analyzer.generate_ai_summary(results)
    print(summary)
    
    # Save results
    analyzer.save_analysis(results, "apple_ai_analysis.json")

if __name__ == "__main__":
    main()