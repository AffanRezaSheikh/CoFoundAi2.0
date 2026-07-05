"""
CV Extractor - Merged version.
Extracts structured information from CV/Resume PDFs.
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import spacy

nlp = spacy.load("en_core_web_sm")


# ==================== PART 1: PDF TO TEXT ====================

def pdf_to_text(pdf_path: str) -> str:
    """Build text line by line from PDF (cleaner than block-based)."""
    text_content = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            text_content.append(line_text.strip())
    return "\n".join(text_content)


# ==================== PART 2: CV EXTRACTOR ====================

class CVExtractor:
    """Extract structured information from CV text."""

    TECH_KEYWORDS = {
        'languages': [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C',
            'Go', 'Rust', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Scala', 'R',
            'MATLAB', 'Julia', 'Dart', 'Perl'
        ],
        'web_frameworks': [
            'React', 'Angular', 'Vue', 'Next.js', 'Nuxt', 'Svelte',
            'Django', 'Flask', 'FastAPI', 'Express', 'Node.js', 'Spring',
            'ASP.NET', 'Laravel', 'Rails', 'Symfony'
        ],
        'ml_ai': [
            'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas',
            'NumPy', 'OpenCV', 'YOLO', 'NLTK', 'spaCy', 'Hugging Face',
            'LangChain', 'XGBoost', 'LightGBM', 'CatBoost'
        ],
        'databases': [
            'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Cassandra',
            'Elasticsearch', 'DynamoDB', 'SQLite', 'Oracle', 'MSSQL'
        ],
        'cloud_devops': [
            'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins',
            'GitHub Actions', 'GitLab CI', 'Terraform', 'Ansible', 'CircleCI'
        ],
        'tools': [
            'Git', 'GitHub', 'GitLab', 'Linux', 'Bash', 'Jupyter',
            'VS Code', 'Postman', 'Jira', 'Confluence'
        ]
    }

    SKILL_ALIASES = {
        'React': ['React', 'ReactJS', 'React.js', 'React Native'],
        'Node.js': ['Node.js', 'NodeJS', 'Node', 'Node JS'],
        'PyTorch': ['PyTorch', 'PyTorch Lightning', 'Torch'],
        'TensorFlow': ['TensorFlow', 'TensorFlow.js', 'TF'],
        'JavaScript': ['JavaScript', 'JS', 'Javascript'],
        'TypeScript': ['TypeScript', 'TS'],
        'Vue': ['Vue', 'Vue.js', 'VueJS'],
        'Angular': ['Angular', 'AngularJS', 'Angular.js'],
        'Next.js': ['Next.js', 'NextJS', 'Next'],
        'Express': ['Express', 'Express.js', 'ExpressJS'],
        'PostgreSQL': ['PostgreSQL', 'Postgres', 'PSQL'],
        'MongoDB': ['MongoDB', 'Mongo'],
        'Scikit-learn': ['Scikit-learn', 'sklearn', 'scikit-learn'],
    }

    NAME_BLACKLIST = {
        'resume', 'cv', 'curriculum vitae', 'profile', 'about me',
        'biodata', 'name', 'contact', 'personal details', 'objective'
    }

    JOB_TITLES = [
        'Senior Software Engineering Intern', 'Software Engineering Intern',
        'Senior Software Engineer', 'Senior Software Developer',
        'Senior Data Scientist', 'Senior Machine Learning Engineer',
        'Senior Backend Developer', 'Senior Frontend Developer',
        'Senior Full Stack Developer', 'Senior DevOps Engineer',
        'Senior Cloud Engineer', 'Senior Mobile Developer',
        'Senior Research Engineer', 'Senior Research Scientist',
        'Senior Product Manager', 'Senior Project Manager',
        'Senior Consultant', 'Senior Data Engineer', 'Senior Data Analyst',
        'Senior QA Engineer', 'Senior Site Reliability Engineer',
        'Machine Learning Engineer', 'Site Reliability Engineer',
        'Solutions Architect', 'Cloud Architect', 'Software Architect',
        'Full Stack Developer', 'Backend Developer', 'Frontend Developer',
        'Mobile Developer', 'iOS Developer', 'Android Developer',
        'Web Developer', 'DevOps Engineer', 'Cloud Engineer',
        'Software Engineer', 'Software Developer',
        'Data Scientist', 'Data Engineer', 'Data Analyst',
        'Research Scientist', 'Research Engineer', 'AI Engineer',
        'ML Engineer', 'NLP Engineer', 'Computer Vision Engineer',
        'QA Engineer', 'Test Engineer', 'Security Engineer',
        'Product Manager', 'Project Manager', 'Technical Lead',
        'Engineering Manager', 'Tech Lead', 'Team Lead',
        'Software Engineering Intern', 'Research Intern', 'Data Science Intern',
        'Machine Learning Intern', 'SDE Intern',
        'Consultant', 'Architect', 'Engineer', 'Developer',
        'Scientist', 'Analyst', 'Intern', 'SDE'
    ]

    SECTION_HEADERS = [
        'summary', 'objective', 'profile', 'about',
        'experience', 'work experience', 'employment', 'professional experience',
        'education', 'academic background', 'qualifications',
        'skills', 'technical skills', 'core competencies', 'expertise',
        'projects', 'personal projects', 'key projects',
        'certifications', 'awards', 'volunteer', 'interests', 'references'
    ]

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split('\n')

    # ---------- 1. CONTACT INFORMATION ----------

    def extract_contact_info(self) -> Dict[str, Optional[str]]:
        return {
            'name': self.extract_name(),
            'email': self.extract_email(),
            'phone': self.extract_phone(),
            'location': self.extract_location(),
            'linkedin': self.extract_linkedin(),
            'github': self.extract_github(),
        }

    def extract_email(self) -> Optional[str]:
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, self.text)
        return match.group() if match else None

    def extract_phone(self) -> Optional[str]:
        patterns = [
            r'\+91[-.\s]?[6-9]\d{9}',
            r'\b[6-9]\d{4}[-.\s]\d{5}\b',
            r'\b[6-9]\d{9}\b',
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        ]
        for pattern in patterns:
            match = re.search(pattern, self.text)
            if match:
                return match.group().strip()
        return None

    def extract_linkedin(self) -> Optional[str]:
        pattern = r'(https?://)?(www\.)?linkedin\.com/in/[\w-]+'
        match = re.search(pattern, self.text, re.IGNORECASE)
        return match.group() if match else None

    def extract_github(self) -> Optional[str]:
        pattern = r'(https?://)?(www\.)?github\.com/[\w-]+'
        match = re.search(pattern, self.text, re.IGNORECASE)
        return match.group() if match else None

    def extract_location(self) -> Optional[str]:
        location_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2}\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z][a-z]+\b',
        ]
        header_text = '\n'.join(self.lines[:10])
        for pattern in location_patterns:
            match = re.search(pattern, header_text)
            if match:
                return match.group()
        return None

    # ---------- 2. NAME EXTRACTION ----------

    def extract_name(self) -> Optional[str]:
        """Top-of-doc heuristic with blacklist, fallback to spaCy NER."""
        for line in self.lines[:8]:
            line = line.strip()
            if not line or '@' in line or re.search(r'\d', line):
                continue
            if line.lower() in self.NAME_BLACKLIST:
                continue
            if line.isupper():
                continue
            words = line.split()
            if not (2 <= len(words) <= 4):
                continue
            if sum(c.isalpha() or c.isspace() for c in line) / len(line) > 0.8:
                return line

        doc = nlp(self.text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) >= 2:
                if ent.text.lower() not in self.NAME_BLACKLIST:
                    return ent.text
        return None

    # ---------- 3. SKILLS EXTRACTION ----------

    def extract_skills(self) -> Dict:
        """Keyword + alias matching across full text (or skills section if found)."""
        section_text = self.extract_section_texts().get('skills', '')
        search_text = section_text if section_text else self.text

        found_skills = {category: [] for category in self.TECH_KEYWORDS.keys()}

        for category, keywords in self.TECH_KEYWORDS.items():
            for skill in keywords:
                variations = self.SKILL_ALIASES.get(skill, [skill])
                for variant in variations:
                    if re.search(rf'\b{re.escape(variant)}\b', search_text, re.IGNORECASE):
                        found_skills[category].append(skill)
                        break

        all_skills = []
        for skills in found_skills.values():
            all_skills.extend(skills)

        return {
            'by_category': found_skills,
            'all_skills': list(set(all_skills)),
            'count': len(set(all_skills))
        }

    # ---------- 4. EDUCATION (section-walking) ----------

    def extract_education(self) -> List[Dict[str, str]]:
        education = []
        in_section = False
        current_entry = {}
        education_headers = ['education', 'academic background', 'qualifications']

        for line in self.lines:
            line_lower = line.lower().strip()

            if any(h == line_lower or line_lower.startswith(h) for h in education_headers):
                in_section = True
                continue

            if in_section and self._is_section_header(line):
                if current_entry:
                    education.append(current_entry)
                break

            if in_section and line.strip():
                date_match = re.search(
                    r'\b(19|20)\d{2}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
                    line
                )
                if date_match:
                    if current_entry:
                        education.append(current_entry)
                        current_entry = {}
                    current_entry['date'] = date_match.group()
                    current_entry['text'] = line.strip()
                elif current_entry:
                    current_entry['text'] += ' ' + line.strip()
                else:
                    current_entry['text'] = line.strip()

        if current_entry:
            education.append(current_entry)
        return education

    # ---------- 5. EXPERIENCE (section-walking) ----------

    def extract_experience(self) -> List[Dict]:
        """Return structured experience entries: role, company, duration, responsibilities."""
        experience_headers = ['experience', 'work experience', 'employment', 'professional experience']

        in_section = False
        section_lines: List[str] = []
        for line in self.lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if any(h == line_lower or line_lower.startswith(h) for h in experience_headers):
                in_section = True
                continue

            if in_section and self._is_section_header(line):
                break

            if in_section:
                section_lines.append(line_stripped)

        return extract_experience_without_gemini('\n'.join(section_lines))

    def extract_experience_years(self) -> Optional[float]:
        patterns = [
            r'(\d+\.?\d*)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience.*?(\d+\.?\d*)\+?\s*years?',
            r'(\d+\.?\d*)\+?\s*yrs?\s+(?:of\s+)?experience'
        ]
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, self.text, re.IGNORECASE)
            years.extend([float(m) for m in matches])
        return max(years) if years else None

    def extract_experience_detailed(self, gemini_api_key: Optional[str] = None) -> List[Dict]:
        experience_section = self._extract_section_content('Experience')

        if not experience_section:
            return []

        # Try local extraction first (free, fast, no API)
        local_result = extract_experience_without_gemini(experience_section)

        # Check quality - if most entries missing role or company, fallback to Gemini
        missing = sum(1 for e in local_result if not e['role'] or not e['company'])

        if gemini_api_key and missing > len(local_result) / 2:
            return self._extract_experience_with_gemini(experience_section, gemini_api_key)

        return local_result

    def _extract_experience_with_gemini(self, section: str, api_key: str) -> List[Dict]:
        try:
            import google.generativeai as genai

            if not section:
                return []

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = (
                "Extract work experience as JSON array with fields: "
                "role, company, duration, responsibilities (list). "
                "Return ONLY valid JSON, no markdown.\n\n" + section
            )
            response = model.generate_content(prompt)

            text = response.text
            start = text.find('[')
            end = text.rfind(']') + 1
            if start == -1 or end == 0:
                return []
            return json.loads(text[start:end])
        except Exception:
            return []

    # ---------- 6. PROJECTS (section-walking) ----------

    def extract_projects(self) -> List[Dict[str, str]]:
        projects = []
        in_section = False
        current_project = {}
        project_headers = ['projects', 'personal projects', 'key projects']

        for line in self.lines:
            line_lower = line.lower().strip()

            if any(h == line_lower or line_lower.startswith(h) for h in project_headers):
                in_section = True
                continue

            if in_section and self._is_section_header(line):
                if current_project:
                    projects.append(current_project)
                break

            if in_section and line.strip():
                if re.match(r'^[•·\-\*]|\d+\.', line.strip()):
                    if current_project:
                        projects.append(current_project)
                        current_project = {}
                    current_project['text'] = line.strip()
                elif current_project:
                    current_project['text'] += ' ' + line.strip()
                else:
                    current_project['text'] = line.strip()

        if current_project:
            projects.append(current_project)
        return projects

    # ---------- 7. ORGANIZATIONS ----------

    def extract_organizations(self) -> List[str]:
        doc = nlp(self.text)
        orgs = [
            ent.text for ent in doc.ents
            if ent.label_ == "ORG"
            and len(ent.text) > 2
            and ent.text not in ['Inc', 'Ltd', 'LLC']
        ]
        return list(set(orgs))

    # ---------- 8. SECTIONS ----------

    def extract_section_texts(self) -> Dict[str, str]:
        sections = {
            'summary': '', 'experience': '', 'projects': '',
            'skills': '', 'education': ''
        }
        section_mapping = {
            'summary': ['summary', 'objective', 'profile', 'about'],
            'experience': ['experience', 'work experience', 'employment', 'professional experience'],
            'projects': ['projects', 'personal projects', 'key projects'],
            'skills': ['skills', 'technical skills', 'core competencies'],
            'education': ['education', 'academic background', 'qualifications']
        }

        current_section = None
        for line in self.lines:
            line_lower = line.lower().strip()

            section_found = False
            for section_name, headers in section_mapping.items():
                if any(h == line_lower or line_lower.startswith(h) for h in headers):
                    current_section = section_name
                    section_found = True
                    break

            if not section_found and current_section and line.strip():
                sections[current_section] += line + '\n'

        return sections

    def _extract_section_content(self, section_name: str) -> str:
        return self.extract_section_texts().get(section_name.lower(), '').strip()

    def _is_section_header(self, line: str) -> bool:
        line = line.strip()
        if len(line) == 0 or len(line) > 50:
            return False
        line_lower = line.lower()
        return any(s == line_lower or line_lower.startswith(s) for s in self.SECTION_HEADERS)

    # ---------- MAIN ----------

    def extract_all(self, gemini_api_key: Optional[str] = None) -> Dict:
        return {
            'contact_info': self.extract_contact_info(),
            'skills': self.extract_skills(),
            'education': self.extract_education(),
            'experience': self.extract_experience(),
            'experience_years': self.extract_experience_years(),
            'experience_detailed': self.extract_experience_detailed(gemini_api_key),
            'projects': self.extract_projects(),
            'organizations': self.extract_organizations(),
            'sections': self.extract_section_texts(),
        }


# ==================== HELPER: LOCAL EXPERIENCE PARSING ====================

def extract_experience_without_gemini(text: str) -> List[Dict]:
    """Parse raw experience-section text into structured entries (no API calls)."""
    DATE_TOKEN = (
        r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
        r'|(?:19|20)\d{2})'
    )
    END_TOKEN = rf'(?:{DATE_TOKEN}|Present|Current)'
    DATE_RANGE_PATTERN = rf'{DATE_TOKEN}\s*[-–—]\s*{END_TOKEN}'

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

    entries: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if re.search(DATE_RANGE_PATTERN, line, re.IGNORECASE):
            if current:
                entries.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append(current)

    job_titles = sorted(CVExtractor.JOB_TITLES, key=len, reverse=True)

    result = []
    for entry_lines in entries:
        header_line = entry_lines[0]

        date_match = re.search(DATE_RANGE_PATTERN, header_line, re.IGNORECASE)
        duration = date_match.group().strip() if date_match else ""

        header_no_date = (
            header_line.replace(date_match.group(), '').strip(' -–—,|\t')
            if date_match else header_line
        )

        role = ""
        for title in job_titles:
            if re.search(rf'\b{re.escape(title)}\b', header_no_date, re.IGNORECASE):
                role = title
                break

        company = ""
        doc = nlp(header_no_date)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                company = ent.text.strip()
                break

        responsibilities = []
        for line in entry_lines[1:]:
            parts = re.split(r'[•·\*\u2022]|(?:^|\s)-\s+', line)
            for part in parts:
                cleaned = part.strip().strip('-').strip()
                if len(cleaned) >= 15:
                    responsibilities.append(cleaned)
                    if len(responsibilities) >= 6:
                        break
            if len(responsibilities) >= 6:
                break

        result.append({
            'role': role,
            'company': company,
            'duration': duration,
            'responsibilities': responsibilities
        })

    return result


# ==================== USAGE ====================

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ").strip().strip('"')

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        raise SystemExit(1)

    text = pdf_to_text(pdf_path)
    print(f"Extracted {len(text.splitlines())} lines from PDF.\n")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    extractor = CVExtractor(text)
    cv_info = extractor.extract_all(gemini_api_key=GEMINI_API_KEY)

    print("=" * 60)
    print("EXTRACTED CV INFORMATION")
    print("=" * 60)

    print("\nCONTACT INFO:")
    for key, val in cv_info['contact_info'].items():
        print(f"  {key.title()}: {val}")

    print(f"\nEXPERIENCE:")
    print(f"  Total Years: {cv_info['experience_years']}")
    print(f"  Companies: {', '.join(cv_info['organizations'][:5])}")

    if cv_info['experience_detailed']:
        print("\nDETAILED EXPERIENCE:")
        for job in cv_info['experience_detailed']:
            if 'role' in job:
                print(f"  - {job.get('role')} at {job.get('company')} ({job.get('duration')})")
                for resp in job.get('responsibilities', [])[:2]:
                    print(f"      * {resp}")
            else:
                print(f"  - {job.get('date', '')}: {job.get('text', '')[:100]}")

    print(f"\nEDUCATION:")
    for edu in cv_info['education']:
        print(f"  - {edu.get('date', '')}: {edu.get('text', '')[:100]}")

    print(f"\nSKILLS ({cv_info['skills']['count']} found):")
    print(f"  {', '.join(cv_info['skills']['all_skills'])}")

    print(f"\nPROJECTS ({len(cv_info['projects'])} found):")
    for proj in cv_info['projects'][:5]:
        print(f"  - {proj.get('text', '')[:100]}")

    print(f"\nSECTIONS FOUND: {', '.join(k for k, v in cv_info['sections'].items() if v)}")

    print("\n" + "=" * 60)

    with open('extracted_cv_data.json', 'w', encoding='utf-8') as f:
        json.dump(cv_info, f, indent=2, ensure_ascii=False)
    print("Saved to extracted_cv_data.json")
