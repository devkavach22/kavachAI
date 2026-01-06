from typing import List, Optional
from pydantic import BaseModel, Field


class Experience(BaseModel):
    title: Optional[str] = ""
    company: Optional[str] = ""
    location: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    responsibilities: Optional[str] = ""


class Education(BaseModel):
    degree: Optional[str] = ""
    institution: Optional[str] = ""
    location: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    gpa_or_percentage: Optional[str] = ""


class TrainingCertificate(BaseModel):
    title: Optional[str] = ""
    institute: Optional[str] = ""


class Project(BaseModel):
    name: Optional[str] = ""
    location: Optional[str] = ""
    start_date: Optional[str] = ""
    end_date: Optional[str] = ""
    description: Optional[str] = ""
    technologies: Optional[List[str]] = Field(default_factory=list)


class Skills(BaseModel):
    development: Optional[List[str]] = Field(default_factory=list)
    database: Optional[List[str]] = Field(default_factory=list)
    version_control: Optional[List[str]] = Field(default_factory=list)
    languages: Optional[List[str]] = Field(default_factory=list)


class PersonalInformation(BaseModel):
    name: Optional[str] = ""
    designation: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
    country: Optional[str] = ""
    dob: Optional[str] = ""
    linkedin: Optional[str] = ""
    summary: Optional[str] = ""


class CVData_formate(BaseModel):
    personal_information: Optional[PersonalInformation] = None
    experience: Optional[List[Experience]] = Field(default_factory=list)
    education: Optional[List[Education]] = Field(default_factory=list)
    training_certificates: Optional[List[TrainingCertificate]] = Field(
        default_factory=list
    )
    projects: Optional[List[Project]] = Field(default_factory=list)
    skills: Optional[Skills] = Skills()  # default empty skills object
