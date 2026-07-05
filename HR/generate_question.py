"""
Generate interview questions from a candidate's CV JSON using Gemini API.
Class-based with fallback safety net for demo reliability.
"""

import os
import json
import time
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

REQUIRED_FIELDS = ["question", "type", "skill_tested", "difficulty", "expected_keywords"]


class QuestionGenerator:
    """
    Generate interview questions from CV data using Gemini API.
    
    Usage:
        gen = QuestionGenerator()  # picks up GEMINI_API_KEY from env
        questions = gen.generate_questions(cv_json, config)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        
        genai.configure(api_key=self.api_key)
        
        # High temperature for varied questions on regenerate
        # Use a "-latest" alias so we don't break when Google retires a
        # specific model snapshot. Override with $GEMINI_MODEL if needed.
        model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.9,
                "max_output_tokens": 2000
            }
        )
    
    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────
    
    def generate_questions(self, cv_json: Dict, config: Dict) -> List[Dict]:
        """
        Generate validated interview questions.
        Falls back to safe generic questions if API fails.
        """
        prompt = self._build_prompt(cv_json, config)
        response_text = self._call_gemini_with_retry(prompt)
        
        if not response_text:
            print("⚠️  API failed - using fallback questions")
            return self._generate_fallback_questions(config)
        
        questions = self._parse_json_response(response_text)
        if not questions:
            print("⚠️  Parsing failed - using fallback questions")
            return self._generate_fallback_questions(config)
        
        valid = self._validate_questions(questions)
        
        # Auto-retry if too many dropped
        if len(valid) < config["num_questions"] // 2 and not config.get("_is_retry"):
            print(f"⚠️  Only {len(valid)} valid - retrying once...")
            return self.generate_questions(cv_json, {**config, "_is_retry": True})
        
        # Final safety net
        if not valid:
            print("⚠️  No valid questions after retry - using fallback")
            return self._generate_fallback_questions(config)
        
        return valid
    
    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────
    
    def _build_prompt(self, cv_json: Dict, config: Dict) -> str:
        # Extract skills with fallback
        skills = cv_json.get("skills", {})
        if isinstance(skills, dict):
            skill_list = ", ".join(skills.get("all_skills", []))
        else:
            skill_list = ", ".join(skills) if isinstance(skills, list) else ""
        
        if not skill_list:
            skill_list = "General software engineering, problem solving"
        
        # Get and truncate other CV fields
        experience = cv_json.get("experience_detailed", cv_json.get("experience", []))
        education = cv_json.get("education", [])
        sections = cv_json.get("sections", {})
        projects = sections.get("projects", "Not provided")
        
        experience_str = json.dumps(experience, indent=2)[:1500]
        education_str = json.dumps(education, indent=2)[:800]
        projects_str = projects[:800] if isinstance(projects, str) else "Not provided"
        
        custom_block = ""
        if config.get("custom_instructions"):
            custom_block = f"\nAdditional instructions: {config['custom_instructions']}"
        
        return f"""
You are a technical interviewer at {config['company_type']}.
You are interviewing a candidate for the role of {config['job_role']}.

INTERVIEW PARAMETERS:
- Difficulty: {config['difficulty']}
- Style: {config['interview_style']}
- Number of questions: {config['num_questions']}
{custom_block}

CANDIDATE PROFILE:
- Skills: {skill_list}
- Experience: {experience_str}
- Education: {education_str}
- Projects: {projects_str}

INSTRUCTIONS:
1. Generate exactly {config['num_questions']} questions
2. Reference the candidate's actual skills and experience - no generic questions
3. Vary question types based on the interview style
4. Make questions progressively harder (start medium, end hard)
5. Include expected keywords for grading reference
6. Tag each question with a broad category for filtering

OUTPUT FORMAT (return ONLY valid JSON array):
[
  {{
    "question": "full question text",
    "type": "technical | experience | behavioral | system_design",
    "category": "DSA | backend | frontend | ml | system_design | devops | behavioral",
    "skill_tested": "specific skill name",
    "difficulty": "easy | medium | hard",
    "expected_keywords": ["keyword1", "keyword2", "keyword3", "keyword4"],
    "follow_up": "optional follow-up question if they answer well"
  }}
]

Do NOT include explanations, markdown, code fences, or any text outside the JSON array.
"""
    
    def _call_gemini_with_retry(self, prompt: str) -> Optional[str]:
        """Call Gemini API with one retry on failure."""
        for attempt in range(2):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt == 1:
                    print(f"⚠️  Gemini API error after retry: {e}")
                    return None
                print(f"⚠️  API call failed (attempt 1), retrying...")
                time.sleep(1)
        return None
    
    def _parse_json_response(self, text: str) -> List[Dict]:
        """Safely extract JSON array from response."""
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start == -1 or end == 0:
                return []
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            return []
    
    def _validate_questions(self, questions: List[Dict]) -> List[Dict]:
        """Drop questions missing required fields."""
        valid = []
        for i, q in enumerate(questions):
            missing = [f for f in REQUIRED_FIELDS if f not in q]
            if missing:
                print(f"⚠️  Question {i+1} missing {missing} - skipping")
                continue
            valid.append(q)
        
        if len(valid) < len(questions):
            print(f"⚠️  Validation: kept {len(valid)}/{len(questions)} questions")
        
        return valid
    
    def _generate_fallback_questions(self, config: Dict) -> List[Dict]:
        """Safety net when API fails - keeps demo running."""
        role = config.get("job_role", "this role")
        return [
            {
                "question": f"Tell me about a project where you applied skills relevant to {role}.",
                "type": "experience",
                "category": "behavioral",
                "skill_tested": "General Experience",
                "difficulty": "easy",
                "expected_keywords": ["project", "team", "implementation", "outcome"],
                "follow_up": "What was the most challenging part?"
            },
            {
                "question": "Describe a challenging technical problem you solved recently.",
                "type": "behavioral",
                "category": "behavioral",
                "skill_tested": "Problem Solving",
                "difficulty": "medium",
                "expected_keywords": ["challenge", "approach", "trade-offs", "result"],
                "follow_up": "How would you approach it differently now?"
            },
            {
                "question": "Walk me through how you'd design a system for a high-traffic application.",
                "type": "system_design",
                "category": "system_design",
                "skill_tested": "System Design",
                "difficulty": "hard",
                "expected_keywords": ["scalability", "load balancing", "database", "caching"],
                "follow_up": "How would your design change at 10x scale?"
            }
        ]


# ════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT (kept from your version)
# ════════════════════════════════════════════════════════════════════

def get_interviewer_config() -> Dict:
    """Collect interview preferences from terminal."""
    print("\n" + "="*60)
    print("INTERVIEWER CONFIGURATION")
    print("="*60)
    
    job_role = input("\nJob role (e.g. Software Engineer): ").strip()
    
    print("\nCompany type:\n  1. Startup\n  2. Mid-size\n  3. Enterprise\n  4. Custom")
    company_choice = input("Choose (1-4): ").strip()
    company_map = {
        "1": "an early-stage startup that values scrappiness and ownership",
        "2": "a mid-size tech company that values technical depth and collaboration",
        "3": "a large enterprise MNC that values structured thinking and scalability"
    }
    company_type = (input("Describe: ").strip() if company_choice == "4" 
                    else company_map.get(company_choice, company_map["2"]))
    
    print("\nDifficulty:\n  1. Junior\n  2. Mid-level\n  3. Senior")
    diff_map = {
        "1": "junior level - fundamentals and learning attitude",
        "2": "mid level - practical experience and code quality",
        "3": "senior level - architecture, trade-offs, and leadership"
    }
    difficulty = diff_map.get(input("Choose (1-3): ").strip(), diff_map["2"])
    
    print("\nStyle:\n  1. Technical deep-dive\n  2. Balanced\n  3. Behavioral focused")
    style_map = {
        "1": "heavily technical with focus on coding and system design",
        "2": "balanced mix of technical depth and behavioral questions",
        "3": "behavioral focused on leadership and culture fit"
    }
    interview_style = style_map.get(input("Choose (1-3): ").strip(), style_map["2"])
    
    num_q = input("\nNumber of questions (default 5): ").strip()
    num_questions = int(num_q) if num_q.isdigit() else 5
    
    custom = input("\nCustom instructions (optional): ").strip()
    
    return {
        "job_role": job_role,
        "company_type": company_type,
        "difficulty": difficulty,
        "interview_style": interview_style,
        "num_questions": num_questions,
        "custom_instructions": custom
    }


def display_questions(questions: List[Dict]):
    if not questions:
        print("\n❌ No questions generated.")
        return
    
    print("\n" + "="*60)
    print(f"GENERATED INTERVIEW QUESTIONS ({len(questions)} total)")
    print("="*60)
    
    for i, q in enumerate(questions, 1):
        print(f"\n┌─ Question {i} " + "─"*45)
        print(f"│ Type      : {q.get('type', 'N/A').upper()}")
        print(f"│ Category  : {q.get('category', 'N/A').upper()}")
        print(f"│ Difficulty: {q.get('difficulty', 'N/A').upper()}")
        print(f"│ Tests     : {q.get('skill_tested', 'N/A')}")
        print("└" + "─"*58)
        print(f"\n  {q.get('question', 'No question text')}")
        
        if q.get('expected_keywords'):
            print(f"\n  Expected keywords: {', '.join(q['expected_keywords'])}")
        if q.get('follow_up'):
            print(f"  Follow-up: {q['follow_up']}")
    
    print("\n" + "="*60)


def save_questions(questions: List[Dict], config: Dict, output_path: str):
    data = {"config": config, "questions": questions, "total_questions": len(questions)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {output_path}")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python question_generator.py <cv_json_path> [output_path]")
        sys.exit(1)
    
    cv_json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "interview_questions.json"
    
    if not os.path.exists(cv_json_path):
        print(f"❌ File not found: {cv_json_path}")
        sys.exit(1)
    
    with open(cv_json_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    
    name = cv_data.get("contact", {}).get("name", "Unknown")
    print(f"\n📄 Loaded CV: {name}")
    
    config = get_interviewer_config()
    
    print("\n⏳ Generating questions via Gemini...")
    generator = QuestionGenerator()
    questions = generator.generate_questions(cv_data, config)
    
    display_questions(questions)
    save_questions(questions, config, output_path)


if __name__ == "__main__":
    main()