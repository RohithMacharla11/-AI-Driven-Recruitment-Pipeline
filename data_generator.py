import random
import pandas as pd
from together import Together
import os
import time

# Configure Together API
os.environ["TOGETHER_API_KEY"] = "c536a4d0073ac5509ddedca17e8850b387550655ae3fd110fc2842eecb6c0600"
client = Together()

# Candidate Names
names =  [
  "Aaron", "Abigail", "Adam", "Adriana", "Aiden", "Alan", "Alexa", "Alice", "Amelia", "Andrew",
  "Angela", "Anna", "Anthony", "Ariana", "Arthur", "Ashley", "Audrey", "Austin", "Ava", "Benjamin",
  "Bethany", "Blake", "Brandon", "Brianna", "Brian", "Brooke", "Caleb", "Cameron", "Carla", "Carlos",
  "Carmen", "Caroline", "Carter", "Catherine", "Chloe", "Christian", "Christina", "Christopher", "Claire",
  "Colin", "Connor", "Courtney", "Daniel", "David", "Derek", "Diana", "Dominic", "Dylan", "Edward",
  "Elena", "Elias", "Elizabeth", "Ella", "Emily", "Emma", "Ethan", "Evan", "Faith", "Felix",
  "Fiona", "Gabriel", "Grace", "Hannah", "Harper", "Hazel", "Henry", "Isabella", "Isaiah", "Jack",
  "Jacob", "James", "Jasmine", "Jason", "Jennifer", "Jessica", "John", "Jonathan", "Joseph", "Joshua",
  "Julia", "Justin", "Kaitlyn", "Katherine", "Kayla", "Kevin", "Kyle", "Liam", "Lily", "Logan",
  "Lucas", "Madeline", "Madison", "Mason", "Matthew", "Megan", "Michael", "Michelle", "Natalie", "Nathan",
  "Nicholas", "Nicole", "Noah", "Nora", "Oliver", "Olivia", "Owen", "Pamela", "Patrick", "Paul",
  "Peter", "Philip", "Piper", "Quinn", "Rachel", "Rebecca", "Riley", "Robert", "Rose", "Ryan",
  "Samantha", "Samuel", "Sarah", "Savannah", "Scott", "Sean", "Sebastian", "Serena", "Shane", "Sienna",
  "Sierra", "Sophia", "Spencer", "Stella", "Stephen", "Steven", "Summer", "Sydney", "Taylor", "Thomas",
  "Tiffany", "Timothy", "Travis", "Trevor", "Trinity", "Tyler", "Vanessa", "Victor", "Victoria", "Vincent",
  "Violet", "Vivian", "Walter", "Wendy", "Weston", "Whitney", "William", "Willow", "Wyatt", "Xavier",
  "Yasmine", "Yvette", "Yvonne", "Zachary", "Zane", "Zara", "Zoe", "Zoey"
]

# Job Roles with Skills and Descriptions
roles_skills = {
    "Data Scientist": {
        "skills": [
            "Statistical analysis", "Machine learning", "Data visualization",
            "Python/R programming", "Big data tools", "SQL", "Feature engineering",
            "Deep learning frameworks", "Cloud platforms", "Communication",
            "Model deployment", "Data storytelling", "Experimentation design",
            "Natural language processing", "Data wrangling"
        ]
    },
    "Software Engineer": {
        "skills": [
            "Python", "Java", "C++", "Data structures", "System design", "Git/GitHub",
            "API development", "Agile methodologies", "Linux", "Cloud services",
            "Code optimization", "Debugging", "Version control", "Microservices",
            "Containerization (Docker)", "Kubernetes", "CI/CD pipelines"
        ]
    },
    "Data Engineer": {
        "skills": [
            "ETL processes", "Big data frameworks", "Cloud data services", "SQL",
            "Programming (Python, Scala)", "Data modeling", "Workflow orchestration tools",
            "Real-time data streaming", "Performance optimization", "API integration",
            "Data warehousing", "Database management", "Pipeline automation",
            "Data security", "Data governance", "Data partitioning"
        ]
    },
    "UI Designer": {
        "skills": [
            "Wireframing", "Prototyping", "Typography", "Responsive design",
            "HTML, CSS", "Graphic design tools", "Design systems", "Usability testing",
            "Accessibility standards", "Information architecture",
            "Color theory", "User flow design", "Interaction design",
            "Animation design", "Branding", "User personas"
        ]
    },
    "Data Analyst": {
        "skills": [
            "Data cleaning", "Statistical analysis", "Data visualization", "SQL",
            "Python/R", "A/B testing", "Presentation skills", "Predictive analytics",
            "Dashboards creation", "Problem-solving", "Business intelligence tools",
            "Data-driven decision-making", "Regression analysis",
            "Hypothesis testing", "Data mining", "Data storytelling"
        ]
    },
    "Product Manager": {
        "skills": [
            "Product lifecycle management", "Market research", "Agile methodologies",
            "Stakeholder management", "Business strategy", "User research",
            "Wireframing tools", "KPI monitoring", "Risk management",
            "Collaboration with teams", "Roadmap development", "Prioritization frameworks",
            "Product marketing", "Competitive analysis", "Customer feedback analysis",
            "Requirement gathering", "Cross-functional leadership"
        ]
    }
}


# Experience Levels and Work Environments
experience_levels = ["Entry-level", "Mid-level", "Senior-level", "Lead", "Director"]
work_environments = ["Remote", "Hybrid", "In-office"]

# Predefined Reasons for Selection and Rejection
selection_reasons = [
    "Excellent problem-solving skills",
    "Strong leadership capabilities",
    "Exceptional communication skills",
    "Extensive experience in related projects",
    "Outstanding project delivery record",
    "Proven track record of innovation",
    "Highly adaptable to new challenges",
    "Expert knowledge of industry tools",
    "Strong customer-oriented approach",
    "Consistently exceeds expectations",
    "Positive team collaboration experience",
    "Ability to meet tight deadlines",
    "Excellent research and development skills",
    "Remarkable technical proficiency",
    "Commitment to continuous learning",
]

rejection_reasons = [
    "Lack of relevant technical experience",
    "Insufficient knowledge of industry tools",
    "Limited problem-solving examples",
    "Minimal team collaboration experience",
    "Incomplete project portfolio",
    "Inconsistent project delivery",
    "Limited leadership experience",
    "Poor communication skills",
    "Lack of innovation in past roles",
    "Failure to meet job-specific requirements",
    "Inadequate understanding of industry trends",
    "Unreliable under tight deadlines",
    "Lack of relevant certifications",
    "Weak customer service experience",
    "Limited technical proficiency in key areas",
]

def generate_result_and_reason():
    result = random.choice(["selected", "rejected"])
    reason = random.choice(selection_reasons if result == "selected" else rejection_reasons)
    return result, reason

def generate_transcript(name, role, result, reason, skills, experience, work_env):
    prompt = (
        f"Simulate a professional interview transcript for {name}, applying for a {role} role. "
        f"The candidate was {result} because of {reason}. Discuss relevant experience ({experience}), key skills ({', '.join(skills)}), "
        f"and their ability to adapt to a {work_env} environment. Make the transcript reflective of real interview dynamics like taking gaps and realistic moments and and there should be no introductory statements or preface."
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_resume(name, role, result, reason, skills, experience, work_env):
    prompt = (
        f"Create a well-formatted resume for {name}, applying for a {role} role. "
        f"The candidate was {result} because of {reason}. Highlight their experience ({experience}), key skills ({', '.join(skills)}), "
        f"and potential to adapt to a {work_env} environment. Ensure the resume is professional and industry-specific and there should be no introductory statements or preface."
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_job_description(role, years):
    prompt = (
        f"Write a job description for a {role} role with at least {years} years of experience. "
        f"Include sections like Job Title, Job Summary, and Key Requirements. Ensure the description is simple(in 1 paragraph not points) not more than 80 words and there should be no introductory statements or preface."
    )
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_random_skills(role):
    skills = roles_skills[role]["skills"]
    return random.sample(skills, min(len(skills), 3))

# Main Automation Loop
data = []
for candidate_id in range(1, 251):
    name = random.choice(names)
    role = random.choice(list(roles_skills.keys()))
    skills = get_random_skills(role)
    result, reason = generate_result_and_reason()
    experience = random.choice(experience_levels)
    work_env = random.choice(work_environments)
    years = random.randint(2, 8)
    transcript = generate_transcript(name, role, result, reason, skills, experience, work_env)
    resume = generate_resume(name, role, result, reason, skills, experience, work_env)
    job_description = generate_job_description(role, years)

    # Append Data
    data.append({
        "ID": f"rohima{candidate_id}",
        "Name": name,
        "Role": role,
        "Transcript": transcript,
        "Resume": resume,
        "Performance (select/reject)": result,
        "Reason for decision": reason,
        "Job Description": job_description,
    })

    # Delay before the next iteration
    time.sleep(3)
    print(f"Processed Candidate {candidate_id}")

# Convert to DataFrame
df = pd.DataFrame(data)


# Save to Excel
df.to_excel("Rohith_Macharla_data.xlsx", index=False)